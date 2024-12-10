import torch
import h5py
from transformers import BertModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

file_path = '/content/ANN-Final-Project/Bert_embeding_datasets/Covid19FakeNews.h5'

# Load input_ids, attention_mask, and labels
with h5py.File(file_path, 'r') as f:
    input_ids = np.array(f['input_ids'])
    attention_mask = np.array(f['attention_mask'])
    labels = np.array(f['labels'])

# Split data
X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=42
)

# Convert to torch tensors
X_train_ids = torch.tensor(X_train_ids, dtype=torch.long)
X_test_ids = torch.tensor(X_test_ids, dtype=torch.long)
X_train_mask = torch.tensor(X_train_mask, dtype=torch.long)
X_test_mask = torch.tensor(X_test_mask, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders for batch processing
train_dataset = TensorDataset(X_train_ids, X_train_mask)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

test_dataset = TensorDataset(X_test_ids, X_test_mask)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# Function to extract embeddings in batches
def extract_embeddings(data_loader, bert_model):
    embeddings = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask = batch
            output = bert_model(input_ids, attention_mask=attention_mask)
            embeddings.append(output.pooler_output.cpu().numpy())
    return np.vstack(embeddings)

# Extract embeddings in batches
X_train_embeddings = extract_embeddings(train_loader, bert_model)
X_test_embeddings = extract_embeddings(test_loader, bert_model)

# Save embeddings and labels to HDF5
output_file = '/content/ANN-Final-Project/Bert_embeding_datasets/Covid19FakeNewsEmbeddings.h5'
with h5py.File(output_file, 'w') as hf:
    hf.create_dataset('train_embeddings', data=X_train_embeddings, compression="gzip")
    hf.create_dataset('test_embeddings', data=X_test_embeddings, compression="gzip")
    hf.create_dataset('y_train', data=y_train.numpy(), compression="gzip")
    hf.create_dataset('y_test', data=y_test.numpy(), compression="gzip")

print("Embeddings saved successfully!")
