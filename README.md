https://www.overleaf.com/6948853616cnzbgdrrjwzb#da7fff

the goal for our project is to use soft prompting in a way that cuts costs while improving model prediction quality. We should try any new ideas we come up.

huggingface prompt-tuning?

The current research I've come accross suggests that the interpretability of soft prompts is a large area of interest. The problem is that while soft prompt vectors can be mapped to the closest tokens in the vocabulary, the resulting projected soft prompts are often arbitrary and nonsensical. Despite this, these nonsensical prompts can still perform well on tasks. 

Waywardness hypothesis - learned soft prompts have the tendency of being nonsensicle and producing nonsensicle results. For example if we ask a model to solve a math question, the learned soft prompt mapped to a vocabulary would be like 'go to the bus' or something. 

Idea - suppose we have a set of text prompts that tend to produce good results when we're inferencing in LLMS. 

The strat is to convert them into the latent space - and with some linear combination of these embbeded prompts, we can produce a new soft prompt that is also interpretable. 


