# World Models, Diffusion, and a Tiny Minecraft World Model

This document explains three things in simple words:

1. What a world model is.

2. What a diffusion model is.

3. Exactly what I built in one night: a tiny Minecraft world that lives inside a neural network.

*You do not need to know deep learning to follow it. Every new word is explained when it first shows up, and there is a small glossary at the end.*

## How to Run This on Your Machine

To build and play the model yourself, run these scripts in order:

1. **`python data_col.py`**

   * First, run this to download the Minecraft videos and action logs. **Note: This is slow, so you will need to wait.**

2. **`python check.py`**

   * Run this to verify the downloaded files are healthy.

3. **`python preprocess.py`**

   * After checking, run this to shrink the pictures and turn the keys into numbers the network can understand.

4. **`python train_diff.py`**

   * Run this to start training the diffusion model.

   * *Tip: It saves one `diff_rollout_XXXX.png` picture every 2000 steps. You can open these to compare them and watch the model learn to draw the world over time.*

5. **`python play_diff.py`**

   * While the training script is running (as soon as a `model_diff.pt` file appears), you can run this to open the window and play the game!

## Part 1: What is a world model?

When you play Minecraft, a real game engine runs. It has code that says: "if the player holds W, move forward. If there is a block in front, stop." Programmers wrote all those rules by hand.

A world model is different. Nobody writes the rules. Instead, a neural network watches a huge amount of gameplay and learns the rules by itself from examples.

After learning, you can give it:

* what the screen looks like right now, and

* what button the player pressed,

and it guesses what the screen will look like next.

That is the whole idea:
**picture now + action = picture next**

If you feed its guess back in as the new "picture now" and keep repeating, you get a game that runs inside the model. There is no Minecraft engine at all. Every frame you see is the network's imagination. Real projects like Oasis (by Decart and Etched) do exactly this with Minecraft.

### Why do people care?

* **Games without engines.** A model could one day generate a whole game just from video.

* **Robots and agents.** If an AI can imagine "what happens if I do this", it can plan ahead safely in its head before acting in the real world.

* **Understanding.** To predict the next frame well, a model has to learn things like "trees stay where they are when I turn around" and "when I walk forward, things get bigger". That is a kind of understanding of how the world works.

### What is hard about it?

* **Guessing the future is uncertain.** If you are walking toward a tree and look behind it, the model never saw what is back there. Many answers are possible.

* **Mistakes pile up.** Each new frame is built from the previous guessed frame, so small errors grow. After a few seconds the world "melts". Even the big professional models struggle with this.

* **No memory.** This tiny model only looks at the last 4 frames. If you turn away and back, it does not remember what was there.

## Part 2: The plan for tonight

I built the project in this order:

1. **Get data:** Minecraft videos plus the keys the player pressed.

2. **Shrink the data:** make the pictures tiny (64 by 64 pixels) so a laptop can handle them.

3. **Version 1:** a simple model that guesses the next frame in one shot. It works, but it is blurry.

4. **Play loop:** a window where you press keys and the model draws the next frame.

5. **Version 2:** a diffusion model that makes sharper pictures.

A small picture of the whole system:

```
Minecraft videos + key logs
        |
        v
  data_col.py  -->  preprocess.py  -->  64x64 frames + action numbers
                                                  |
                                                  v
                                          train model (version 1 or 2)
                                                  |
                                                  v
                                  play.py: your keys --> model --> next frame

```

## Part 3: The data

### Where it comes from

I used the VPT dataset from OpenAI. VPT stands for "Video PreTraining". It is a big collection of videos of real people playing Minecraft. Each video comes with a second file that lists, for every single moment, which keys and mouse movements the player used.

That pairing is what makes it perfect for a world model:

* the video gives us the pictures

* the log file gives us the actions

Each video is about 5 minutes long and runs at 20 frames per second. That means about 6,000 frames per video.

### Downloading (data_col.py)

The script does these steps:

1. Downloads a small "index" file. It is a list of all the videos and where they live on the server.

2. For the first N videos, downloads the `.mp4` (the video) and the `.jsonl` (the action log).

3. A `.jsonl` file is a text file where every line is one small piece of data. Here, one line equals one moment in the game.

### Problems I hit (and what they teach)

| Problem | What happened | Fix | 
| ----- | ----- | ----- | 
| `.mp4.mp4` and a 404 error | The index already included `.mp4` in each path, and the script added it again | Remove the extension first, then add my own | 
| It looked frozen | The download had no progress display, and files are 100+ MB | Added a progress counter | 
| A broken video ("moov atom not found") | A download stopped halfway but the half file looked complete | Download to a `.part` file first, and rename it only when it is finished | 
| Timeout error | The server paused for 30 seconds and the script gave up | Added retries with a longer timeout | 

The lesson: when downloading big files, always write to a temporary name, show progress, and retry on failure.

I ended with 6 good training videos, about 36,000 frames.

## Part 4: Preparing the data (preprocess.py)

Neural networks need numbers, and a laptop needs small images. So I did two conversions.

### Shrinking pictures

Every frame from the video (640 by 360) is resized to 64 by 64 pixels. That throws away detail, but it makes training thousands of times cheaper. This is why the results look blocky and small. It is a tradeoff on purpose.

### Turning actions into numbers

Each moment's keys become a list of 10 numbers:

| Position | Meaning | 
| ----- | ----- | 
| 1 | W pressed (1 or 0) | 
| 2 | A pressed | 
| 3 | S pressed | 
| 4 | D pressed | 
| 5 | Space (jump) | 
| 6 | Shift | 
| 7 | Ctrl | 
| 8 | Mouse moved sideways (scaled to between -1 and 1) | 
| 9 | Mouse moved up or down (scaled to between -1 and 1) | 
| 10 | Left click | 

Everything is saved into `.npz` files. Each file has frames with shape `(N, 64, 64, 3)`, meaning N pictures of 64 by 64 pixels with 3 colors (red, green, blue), and actions with shape `(N, 10)`.

Frame number $t$ and action number $t$ line up. The model learns: "given these pictures and this action, what is picture $t+1$?"

## Part 5: Version 1, the simple model

### What it looks at and what it guesses

* **Input:** the last 4 frames (so it can sense movement, not just a still picture) and the current action.

* **Output:** the next frame.

### The network shape

The network is a small U-Net. A U-Net is a common shape for image-to-image tasks:

* It first shrinks the picture step by step, so it can think about the big picture ("there is a tree on the left").

* Then it grows it back to full size, and uses shortcuts from the early layers so fine details are not lost.

The action numbers are injected at a few places in the middle of the network, so the network can change its guess based on the key you pressed.

It has about 3.4 million parameters. (Parameters are the numbers inside the network that get adjusted during learning. Big modern models have billions, so this one is tiny.)

### A clever trick: predict the change, not the picture

Instead of drawing the whole next frame from nothing, the network predicts only what changes and adds that to the last frame. Most of the screen stays the same from one frame to the next, so this is much easier.

### How it learned

Training works like this, repeated 8,000 times:

1. Pick a random batch of 64 moments from the videos.

2. Show the model the 4 frames and the action.

3. Compare its guess with the real next frame. The difference is called the loss. I used MSE (mean squared error), which just measures how far each pixel is from the right color.

4. Nudge all the parameters a tiny bit so the loss gets smaller.

I also added a tiny bit of random noise to the input frames while training. This teaches the model to survive small imperfections, which will matter when it feeds on its own guesses.

### How I knew it was learning

A good test needs a baseline. Mine was a lazy trick: just copy the last frame and say that is the next one. If the model cannot beat that, it has learned nothing.

I also kept one video aside that the model never trained on, to check it on new material.

| Moment | Model score | Copy-last-frame score | Model is better by | 
| ----- | ----- | ----- | ----- | 
| Step 2000 | 0.00501 | 0.00776 | about 35% | 
| Step 8000 | 0.00415 | 0.00704 | about 41% | 

Lower is better. The model clearly learned something, and more training helped.

One more clue: training loss was around 0.001 but the check on the unseen video was around 0.004. That gap means the model does better on videos it has seen. With only 3 training videos at the time, that is normal. The fix is more data, which is why I kept downloading and later trained on 5 videos.

## Part 6: Why Version 1 looks blurry

When you looked at the rollout pictures (top row real, bottom row model), the first guess looked fine and then everything turned into a dark, dreamy smudge. Two separate reasons:

* **Reason 1: Blurry is the safe guess.** Imagine the model is unsure what is behind a tree. It could be grass, or a cave, or sky. If it picks one and is wrong, the MSE score is terrible. If it blends all the possibilities together, it is never very wrong. So it learns to blend. The blend of many sharp pictures is a blurry picture. This is built into the method. More data and more training help only a little.

* **Reason 2: Mistakes pile up.** Only the first predicted frame comes from real pictures. After that, the model reads its own blurry output to make the next one. Each step adds more blur and more errors, like a photocopy of a photocopy of a photocopy.

This is not a bug. It is how this kind of model behaves, and it proved that the whole pipeline (data, training, looping) works.

## Part 7: Playing it (play.py)

The play script loads the trained model and runs a loop 20 times per second:

1. Read which keys you are holding.

2. Turn them into the 10 action numbers.

3. Ask the model for the next frame.

4. Show it in a window.

5. Add that frame to the model's memory of the last 4 frames, and repeat.

**Controls:** `W A S D` to move, `J` and `L` to look left and right, `I` and `K` to look up and down, `Space` to jump, `C` to click, `R` to reset, `Q` to quit.

### The window problem

The first version used OpenCV to open the window. On your Linux setup (Nobara with Wayland) it crashed with a "Qt platform plugin" error. The model was fine. Only the window tool could not connect to the screen. I switched to pygame, which worked, and which also reads held keys more smoothly.

If pygame ever fails, you can force the display type:

```
SDL_VIDEODRIVER=wayland python play.py
SDL_VIDEODRIVER=x11 python play.py

```

## Part 8: What is a diffusion model?

This is the upgrade that makes pictures sharp. Here is the idea with an everyday picture.

### The sculpture idea

Imagine a block of TV static (random colored dots). Now imagine a very skilled artist who can clean up static a tiny bit at a time. After one cleanup, a rough shape appears. After a few more, you can see a tree. After enough, you have a clear picture.
That artist is the diffusion model.

### Why is it sharper?

Remember the blur problem: the old model blended all possible answers. A diffusion model starts from random noise, and the noise makes the choice for it. Different noise leads to a different specific, sharp picture. It commits to one answer instead of averaging them all.

This is the method behind most modern image generators, and behind real Minecraft world models like Oasis.

### How training works (this version)

For each training example I have a real next frame, call it Y. I also make a picture of pure random noise, call it Z. Then:

1. Pick a random number $t$ between 0 and 1.

2. Mix them: the mixed picture is mostly noise when $t$ is near 0, and mostly the real picture when $t$ is near 1.

3. Ask the model: "Looking at this mixed picture, the last 4 frames, the action, and how noisy it is, which direction should I move to get toward the real picture?"

4. The correct answer is simply Y minus Z (the straight line from the noise to the real picture).

5. Measure how wrong the model was and nudge its parameters.

So the model learns to be a "cleanup guide" at every noise level.

### How it makes a frame (sampling)

To make the next frame in the game:

1. Start with pure random noise.

2. Ask the model which way to move. Take a small step in that direction.

3. Repeat. I used 6 to 8 steps.

The result is the next frame. The model gets to look at the last 4 real (or previously generated) frames and your action at every step, so the new picture fits the scene and your key press.

*Tradeoff:* every frame now needs 6 to 8 model runs instead of 1. So it plays slower. Fewer steps means faster but rougher pictures.

### Extra tricks I added

* **Noise on the old frames.** A tiny bit of noise is added to the 4 context frames, in training and while playing. This helps the model recover when its own earlier frames are slightly off.

* **EMA copy of the model.** EMA stands for exponential moving average. I keep a second copy of the model whose numbers move slowly, like a smoothed average of the training copy. It usually gives cleaner pictures. The saved file `model_diff.pt` is this smoothed copy.

## Part 9: The three files of Version 2

| File | What it does | 
| ----- | ----- | 
| `diffnet.py` | Holds the network, which is a U-Net like before but it also receives the noisy picture and the "how noisy" number. Also holds the sample function that turns noise into a frame. | 
| `train_diff.py` | Loads the data, trains for 20,000 steps, and every 2,000 steps saves a picture (`diff_rollout_XXXX.png`) and the model (`model_diff.pt`). | 
| `play_diff.py` | The same play window as before, but each frame comes from the diffusion sampler. Change `SAMPLE_STEPS` to trade speed against quality. | 

The network, training, and playing all share `diffnet.py`, so they can never disagree about the model's shape.

How to run: first `python train_diff.py`. It takes maybe 30 to 60 minutes. Look at `diff_rollout_2000.png`, then 4000, and so on. Early pictures look like mush, and that is normal. By around 8,000 to 10,000 steps they should be sharper than Version 1. You can start `python play_diff.py` as soon as `model_diff.pt` appears.

## Part 10: Honest limits

It is important to know what this project is and is not.

* **It is tiny.** 64 by 64 pixels, a few million parameters, 6 videos. Real systems use hundreds of times more of everything.

* **Sharp does not mean correct.** Diffusion invents plausible details. A texture can look right without matching what was really there.

* **It still drifts.** After several seconds the scene wanders away from realistic. Press R to reset.

* **No long-term memory.** If you turn around, the place behind you may be different. The model only remembers the last 4 frames.

* **It might ignore your keys a bit.** A good test: hold J or L and see if the picture slides sideways. If it barely reacts, the model learned to just copy frames, and I need more data or training on its own mistakes.

## Part 11: What to try next

From easiest to hardest:

* **More videos.** This is the best single upgrade. Keep the downloader running and retrain.

* **Train on its own mistakes.** While training, make the model predict 3 or 4 frames in a row using its own outputs, so it learns to recover from errors.

* **Train longer.** Raise `STEPS` in `train_diff.py` and leave it running overnight.

* **Bigger pictures.** Go to 96 by 96 or 128 by 128 once everything works.

* **Longer memory.** Feed more than 4 previous frames, or add a simple memory of the scene.

* **Faster sampling.** Use fewer steps or smarter samplers so it plays smoother.

## Part 12: Quick troubleshooting

| Symptom | Likely cause | What to do | 
| ----- | ----- | ----- | 
| Download seems frozen | Big files, no progress shown | Use the progress version of the script, check the file size is growing | 
| "moov atom not found" | Broken partial video | Delete that `.mp4` and its `.npz`, run the downloader again | 
| Timeout while downloading | Server or network hiccup | Run the script again, it skips finished files and retries | 
| Qt or display error in `play.py` | OpenCV cannot talk to Wayland | Use the pygame version | 
| Picture melts quickly | Normal for small models | Press R. Fix with more data and training on its own outputs | 
| Diffusion game is laggy | Too many sample steps | Lower `SAMPLE_STEPS` | 
| Model ignores your keys | Learned to copy frames | More data, longer training, or train with multi-step rollouts | 

## Glossary

* **World model:** a network that predicts the next picture from the current picture and an action.

* **Frame:** one picture from a video. The videos run at 20 frames per second.

* **Action:** what the player did (keys and mouse), turned into numbers.

* **Neural network:** a big math function with adjustable numbers, which learns from examples.

* **Parameters:** those adjustable numbers inside the network.

* **Training:** repeatedly showing examples and nudging the parameters to reduce mistakes.

* **Loss:** a score for how wrong the model was. Lower is better.

* **MSE:** mean squared error, the loss that measures how far each pixel is from the right color.

* **Baseline:** a simple trick to compare against (mine was copying the last frame).

* **Validation:** checking the model on data it never trained on.

* **U-Net:** a network shape that shrinks the picture, then grows it back, with shortcuts to keep details.

* **Rollout:** running the model on its own outputs for many frames in a row.

* **Diffusion model:** a model that makes a picture by cleaning random noise step by step.

* **Sampling:** the process of generating a frame from noise.

* **EMA:** a slowly moving average copy of the model, which usually gives cleaner results.

* **VPT:** OpenAI's dataset of Minecraft gameplay videos with action logs.

* **GPU:** the graphics chip that makes training fast. I used CUDA, which is how Python talks to an NVIDIA GPU.


## by Paxton.