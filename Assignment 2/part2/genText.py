import torch
import numpy as np

def generate_text(model, dataset, length, device, stochastic = False):
    inp_original = 'a'
    ind_list = range(dataset.vocab_size)
    while inp_original != '!':
        print()
        print("Type a string to generate text starting with it, then press enter. (type ! to exit)")
        inp_original = input()
        inp = torch.tensor([]).to(device)
        for char in inp_original:
            inp_new = torch.LongTensor([[dataset._char_to_ix[char]]]).to(device)
            inp_new = torch.nn.functional.one_hot(inp_new, dataset.vocab_size).float()
            inp = torch.cat((inp, inp_new), dim=1)
        inp_new = ''
        for i in range(length):
            if inp_new != '':
                
                inp_new = torch.LongTensor([[dataset._char_to_ix[inp_new]]]).to(device)
                inp_new = torch.nn.functional.one_hot(inp_new, dataset.vocab_size).float()
                inp = torch.cat((inp, inp_new), dim=1)
            if stochastic:
                ## apply stochastic sampling on the outputs

                o = torch.nn.functional.softmax(model(inp)[:, -1, :], dim=1)
                o = o.cpu().detach().numpy().squeeze()
                o_char = np.random.choice(ind_list, p=o)
                o_char = dataset._ix_to_char[o_char]
            else:
                o_char = dataset._ix_to_char[torch.argmax(model(inp)[:, -1, :]).item()]
            print(o_char, end='')
            inp_new = o_char

