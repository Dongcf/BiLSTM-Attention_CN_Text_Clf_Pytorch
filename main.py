import os
import argparse
import sys
import torch
import torch.nn.functional as F
import torchtext.data as data
from torchtext.vocab import Vectors
import torch.optim as optim
import dataset
from bi_lstm_attention import  AttentionModel


# Training settings
parser = argparse.ArgumentParser(description="Pytorch LSTM + Attention text classification")
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-momentum',type=float,default=0.9,help='sgd momentum [default: 0.9]')
parser.add_argument('-l2',type=float,default=0,help='weight decay L2 constraint of parameters [default: 0.7]')
parser.add_argument('-dropout', type=float, default=0.3, help='the probability for dropout [default: 0.5]')
parser.add_argument('-batch-size',type=int,default=64,help='batch size for training [default: 128]')
parser.add_argument('-test-batch-size', type=int, default=1024,help='input batch size for testing (default: 1024)')
parser.add_argument('-epochs', type=int, default=16, help='number of epochs for train [default: 20]')
parser.add_argument('-optim',type=str,default='adam',help='the optimizer methods [sgd,momentum,RMSprop,adam]')
parser.add_argument('-layer-size',type=int,default=1,help='the network layer [default 1]')
parser.add_argument('-bidirectional',type=bool,default=True,help='whether to use bidirectional network [default False]')
parser.add_argument('-hidden-size',type=int,default=256,help='number of hidden size for one rnn cell [default: 256]')
parser.add_argument('-embed-dim',type=int,default=300,help='number of embedding dimension [default: 128]')
parser.add_argument('-attention-size',type=int,default=16,help='attention size [default:16]')

# infomation output
parser.add_argument('-device', type=int, default=1, help='device to use for iterate data, -1 mean cpu,1 mean gpu [default: -1]')
parser.add_argument('-log-interval', type=int, default=1,help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=70,help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=1000,help='iteration numbers to stop without performance increasing')

# model save/pre_train path
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-static', type=bool, default=False, help='whether to use static pre-trained word vectors')
parser.add_argument('-pretrained-name', type=str, default=None,help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default=None, help='path of pre-trained word vectors')
parser.add_argument('-sequence-length',type=int,default=20,help='the length of input sentence [default 16]')
#parser.add_argument('-num-classes',type=int,default=2,help='number of classification [default: 2]')
#parser.add_argument('-vocab-size',type=int,default=vocab_size,help='number of vocabulary size')
#parser.add_argument('-embedding-weights',type=float,default=word_embeddings,help='embedding weights')
args = parser.parse_args()

def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    train_dataset, dev_dataset, test_dataset = dataset.get_dataset('data', text_field, label_field)
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_dataset, dev_dataset,test_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset),len(test_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter, dev_iter, test_iter


print('Loading data...')
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
args.vocab_size = len(text_field.vocab)
args.output_size = len(label_field.vocab)
args.cuda = args.device != -1 and torch.cuda.is_available()


print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))

#
def clip_gradient(model,clip_value):
	params = list(filter(lambda p:p.grad is not None,model.parameters()))
	for p in params:
		p.grad.data.clamp_(-clip_value,clip_value)

def save(model, save_dir, save_prefix, epoch,steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}_steps_{}.pt'.format(save_prefix,epoch, steps)
    torch.save(model.state_dict(), save_path)
    return save_path

def train_model(model,train_iter, dev_iter, args):
	if args.cuda:
		model.cuda()
	if args.optim == 'sgd':
		optimizer = optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.l2)
	elif args.optim == 'momentum':
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2,nesterov=True)
	elif args.optim == 'RMSprop':
		optimizer = optim.RMSprop(model.parameters(),lr=args.lr,alpha=0.99,eps=1e-8,weight_decay=args.l2)
	else:
		optimizer = optim.Adam(model.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=args.l2)
	best_acc = 0
	last_step = 0
	for epoch in range(1, args.epochs + 1):
		steps = 0
		for batch in train_iter:
			feature, target = batch.text, batch.cluster_name
			with torch.no_grad():
				feature.t_(), target.sub_(1)
			if args.cuda:
				feature, target = feature.cuda(), target.cuda()
			optimizer.zero_grad()
			logits = model(feature)
			loss = F.cross_entropy(logits, target)
			loss.backward()
			# clip_gradient(model, 1)
			optimizer.step()
			if steps % args.log_interval == 0:
				corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
				train_acc = 100.0 * corrects / batch.batch_size
				sys.stdout.write('\rEpoch[{}] - Step[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,steps,loss.item(),
					                                                         train_acc,corrects,batch.batch_size))
			steps += 1
			if steps % args.test_interval == 0:
				dev_acc = eval_model(model,dev_iter,  args)
				if dev_acc > best_acc:
					best_acc = dev_acc
					last_step = steps
					if args.save_best:
						print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
						saved_model_name = save(model, args.save_dir, 'best',epoch, steps)
				else:
					if steps - last_step >= args.early_stopping:
						print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
						raise KeyboardInterrupt
				model.train()
	return  saved_model_name
def eval_model(model,data_iter,args):
	size,corrects, avg_loss = 0, 0,0
	model.eval()
	for batch in data_iter:
		feature, target = batch.text, batch.cluster_name
		with torch.no_grad():
			feature.t_(), target.sub_(1)
		# feature.data.t_(), target.data.sub_(1)
		if args.cuda:
			feature, target = feature.cuda(), target.cuda()
		logits = model(feature)
		loss = F.cross_entropy(logits, target)
		avg_loss += loss.item()
		corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
		size += batch.batch_size
	avg_loss /= size
	accuracy = 100.0 * corrects / size
	print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss,accuracy,corrects,size))
	return accuracy

model = AttentionModel(args)
model_name = train_model(model,train_iter,dev_iter,args)
print('\n'+model_name)
model = AttentionModel(args)
if args.cuda:
	model.cuda()
model.load_state_dict(torch.load(model_name))
test_acc = eval_model(model,test_iter,args)
print(f'Test Acc: {test_acc:.2f}%')

