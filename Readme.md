# Few Shot Learning Reinforcement Learning for GPT models

We first take a small model. In the first itteration of the software we will run [ElutherAI's GPT neo 125M](https://huggingface.co/EleutherAI/gpt-neo-125M). We train a classifier using [SetFit](https://huggingface.co/blog/setfit) to guide the output to what we want. Then we use the classifier to use Reinforcement learning to train the original model to generate helpful text. 

A summery of what I am trying to do is from [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf).

Steps:

1. Train an autoregressive model on large corpus (fresh and relevant to QA datasets)
2. Train supervisedly on selected QA datasets
3. Train a reward model which will judge answers by prompts
4. Train a PPO-guided model from fine-tuned model
5. Repeat the loop

Note: this has been achieved somehow at [this notebook](https://github.com/James4Ever0/FewShotRLGPT/blob/main/textrl-rlhf-chatgpt-check-vram-usage.ipynb). The primary goal is to set a proper reward model, which will faithfully judge answers according to prompt and relevance (experienced! better be gpt-like instead of bert or sentence transformer).

I only find a few repos useful and others daunting (might still useful in some way!).

Namely, they are:

[TextRL](https://github.com/voidful/TextRL)

[minRLHF](https://github.com/thomfoster/minRLHF)
