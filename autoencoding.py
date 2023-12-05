import torch
import seaborn as sns
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = load_dataset('bigscience/P3', 'cos_e_v1.11_aligned_with_common_sense')
train_dataset = dataset['train']

# Initialize the tokenizer and model
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#Learn the latent space
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, input_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)
    
    
latent_dim = 128
d = model.model.shared.embedding_dim
batch_size = 8
epochs = 10 

autoencoder = Autoencoder(d, latent_dim).to(device)
autoencoder_optimizer = AdamW(autoencoder.parameters())

print('Training Autoencoder')
for epoch in range(epochs):
    epoch_loss = 0
    print(f'Epoch {epoch + 1}/{epochs}')
    for i in range(0, len(train_dataset), batch_size):
        batch = train_dataset[i:i+batch_size]
        input_ids = tokenizer(batch['inputs_pretokenized'], return_tensors='pt', padding=True, truncation=True).input_ids
        input_embeddings = model.model.shared(input_ids.to(device))
        latent_representations = autoencoder(input_embeddings)
        autoencoder_loss = torch.mean((latent_representations - input_embeddings) ** 2)
        print(f'\r complete from this epoch {i}/{len(train_dataset)} with loss {autoencoder_loss}', end='')
        epoch_loss += autoencoder_loss.item()
        autoencoder_loss.backward()
        autoencoder_optimizer.step()
        autoencoder_optimizer.zero_grad()
    print(f'Epoch {epoch + 1} Loss: {epoch_loss}')  # print average loss per epoch

# Select a sample
sample = train_dataset[0]

# Tokenize the sample and get input embeddings
input_ids = tokenizer(sample['inputs_pretokenized'], return_tensors='pt').input_ids
input_embeddings = model.model.shared(input_ids.to(device))

# Pass the input embeddings through the autoencoder
reconstructed_embeddings = autoencoder(input_embeddings)

# Compare the input embeddings and the reconstructed embeddings
comparison = torch.mean((reconstructed_embeddings - input_embeddings) ** 2)
print(f'Comparison: {comparison.item()}')

# Freeze the model parameters
# for param in model.parameters():
#     param.requires_grad = False

# # Define the soft prompt
# L = 20
# d = model.model.shared.embedding_dim
# soft_prompt = torch.nn.Parameter(torch.randn(L, d))
# optimizer = AdamW([soft_prompt])

# # Training parameters
# epochs = 1
# batch_size = 8
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# soft_prompt.to(device)

# print('starting training')

# # Training loop
# losses = []
# for epoch in range(epochs):
#     epoch_loss = 0
#     for i in range(0, len(train_dataset) - 9000, batch_size):
#         batch = train_dataset[i:i+batch_size]
#         input_ids = tokenizer(batch['inputs_pretokenized'], return_tensors='pt', padding=True, truncation=True).input_ids
#         labels = tokenizer(batch['targets_pretokenized'], return_tensors='pt', padding=True, truncation=True).input_ids 
#         input_ids = input_ids.to(device)
#         labels = labels.to(device)

#         # Get the input embeddings
#         input_embeddings = model.model.shared(input_ids)

#         soft_prompt_batch = soft_prompt.unsqueeze(0).repeat(input_embeddings.size(0), 1, 1).to(device)
#         combined_embeddings = torch.cat([soft_prompt_batch, input_embeddings], dim=1)

#         # Pass the combined embeddings through the model
#         outputs = model(inputs_embeds=combined_embeddings, labels=labels)
#         print(f'input embeddings shape: {input_embeddings.shape}')
#         print(f'combined embeddings shape: {combined_embeddings.shape}')
#         print(f'outputs shape: {outputs.logits.shape}')
#         print(f'labels shape: {labels.shape}')
#         print(f'labels: {labels}')
#         print(f'labels pretokenized: {batch["targets_pretokenized"]}')
#         loss = outputs.loss
#         epoch_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         print(f'\r complete from this epoch {i}/{len(train_dataset)}', end='')
#         print(f'\r loss: {loss.item()}', end='')


#     epoch_loss /= len(train_dataset)
#     losses.append(epoch_loss)
#     print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

# # Create a DataFrame with epoch and loss data
# loss_data = pd.DataFrame({'Epoch': range(1, epochs + 1), 'Loss': losses})

# # Plot loss over time using Seaborn
# sns.lineplot(data=loss_data, x='Epoch', y='Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()