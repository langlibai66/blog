# DL实战

## Word2Vec

### CBOW

```python
class CBOW(nn.Module):
	def __init__(self,vacab_size,embed_dim):
		super(CBOW,self).__init__()
		self.input_embedding = nn.Embedding(vocab_size,embed_dim)
		self.output_embedding = nn.Embedding(vocab_size,embed_dim)
		nn.init.normal(self.input_embedding.weight.data,mean=0,std=0.01)
		nn.init.normal(self.output_embedding.weight.data,mean=0,std0.01)
		self.loss_fn = nn.BCEWithLogitsLoss()
	def forward(self,input):
    	context = inputs['contexts']
    	target = inputs['targets']
    	label = inputs['labels']
    	context_embedding = self.input_embedding(context)
    	context_embedding = context_embedding.mean(1,keepdim=True)
    	
    	target_embedding = self.output_embedding(target)
    	embedding = context_embedding * target_embedding
    	embedding = torch.sum(embedding,dim=2)
    	loss = self.loss_fn(embedding,label.float())
    	return loss
```

## Skip_Gram

```python
class Skip_gram(nn.Module):
    def __init__(self,vocab_size,embed_dim)
        super(Skip,self).__init__()
    	self.input_embedding = nn.Embedding(vocab_size,embed_dim)
    	self.output_embedding = nn.Embedding(vocab_size,embed_dim)
    	nn.init_normal(self.input_embedding.weight.data,mean=0,std=0.01)
        nn.init_normal(self.output_embedding.weight,data,mean=0,std=0.01)
        nn.lossfn = nn.BCEWithLogitsLoss()
	def forward(self,inputs):
        center_ids = inputs['center_ids']
        context_ids = inputs['context_ids']
        label = inputs['labels']
        center_embedding = self.input_embedding(centet_ids)
        center_embedding = center_embedding.unsqueeze(1)
        
        context_embedding = self.output_embedding(context_ids)
        
        embedding = center_embedding * context_embedding
        embedding = torch.sum(embedding,dim=2)
        
        loss = self.loss_fn(embedding,label.float())
        
        return loss
```

