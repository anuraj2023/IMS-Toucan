import json
import os
import time

import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader

from PreprocessingForTTS.ProcessText import TextFrontend
from Utility.WarmupScheduler import WarmupScheduler


def plot_progress_spec(net, device, save_dir, step, lang, reference_spemb_for_plot):
    tf = TextFrontend(language=lang,
                      use_panphon_vectors=False,
                      use_word_boundaries=False,
                      use_explicit_eos=False)
    sentence = "Hello"
    if lang == "en":
        sentence = "Many animals of even complex structure which " \
                   "live parasitically within others are wholly " \
                   "devoid of an alimentary cavity."
    elif lang == "de":
        sentence = "Dies ist ein brandneuer Satz, und er ist noch dazu " \
                   "ziemlich lang und komplex, dmait man im Spektrogram auch was sieht."
    text = tf.string_to_tensor(sentence).long().squeeze(0).to(device)
    spec = net.inference(text=text, spembs=reference_spemb_for_plot).to("cpu").numpy()
    if not os.path.exists(os.path.join(save_dir, "spec")):
        os.makedirs(os.path.join(save_dir, "spec"))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    lbd.specshow(spec, ax=ax[0][0], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec"), str(step) + ".png"))
    plt.clf()
    plt.close()


def collate_and_pad(batch):
    if len(batch[0]) == 7:
        # every entry in batch: [text, text_length, spec, spec_length, durations, energy, pitch]
        texts = list()
        text_lens = list()
        speechs = list()
        speech_lens = list()
        durations = list()
        pitch = list()
        energy = list()
        for datapoint in batch:
            texts.append(torch.LongTensor(datapoint[0]).squeeze(0))
            text_lens.append(torch.LongTensor([datapoint[1]]))
            speechs.append(torch.Tensor(datapoint[2]))
            speech_lens.append(torch.LongTensor([datapoint[3]]))
            durations.append(torch.LongTensor(datapoint[4]))
            energy.append(torch.Tensor(datapoint[5]))
            pitch.append(torch.Tensor(datapoint[6]))
        return (pad_sequence(texts, batch_first=True),
                torch.stack(text_lens).squeeze(1),
                pad_sequence(speechs, batch_first=True),
                torch.stack(speech_lens).squeeze(1),
                pad_sequence(durations, batch_first=True),
                pad_sequence(pitch, batch_first=True),
                pad_sequence(energy, batch_first=True))
    elif len(batch[0]) == 8:
        # every entry in batch: [text, text_length, spec, spec_length, durations, energy, pitch, spemb]
        texts = list()
        text_lens = list()
        speechs = list()
        speech_lens = list()
        durations = list()
        pitch = list()
        energy = list()
        spembs = list()
        for datapoint in batch:
            texts.append(torch.LongTensor(datapoint[0]).squeeze(0))
            text_lens.append(torch.LongTensor([datapoint[1]]))
            speechs.append(torch.Tensor(datapoint[2]))
            speech_lens.append(torch.LongTensor([datapoint[3]]))
            durations.append(torch.LongTensor(datapoint[4]))
            energy.append(torch.Tensor(datapoint[5]))
            pitch.append(torch.Tensor(datapoint[6]))
            spembs.append(torch.Tensor(datapoint[7]))
        return (pad_sequence(texts, batch_first=True),
                torch.stack(text_lens).squeeze(1),
                pad_sequence(speechs, batch_first=True),
                torch.stack(speech_lens).squeeze(1),
                pad_sequence(durations, batch_first=True),
                pad_sequence(pitch, batch_first=True),
                pad_sequence(energy, batch_first=True),
                torch.stack(spembs))  # may need squeezing, cannot test atm


def train_loop(net, train_dataset, valid_dataset, device, save_directory,
               config, batchsize=32, epochs=150, gradient_accumulation=1,
               epochs_per_save=10, spemb=False, lang="en", lr=0.1, warmup_steps=14000):
    """
    :param lang: language of the synthesis
    :param spemb: whether to expect speaker embeddings
    :param net: Model to train
    :param train_dataset: Pytorch Dataset Object for train data
    :param valid_dataset: Pytorch Dataset Object for validation data
    :param device: Device to put the loaded tensors on
    :param save_directory: Where to save the checkpoints
    :param config: Config of the model to be trained
    :param batchsize: How many elements should be loaded at once
    :param epochs: how many epochs to train for
    :param gradient_accumulation: how many batches to average before stepping
    :param epochs_per_save: how many epochs to train in between checkpoints
    """
    net = net.to(device)
    scaler = GradScaler()
    if spemb:
        reference_spemb_for_plot = torch.Tensor(valid_dataset[0][7]).to(device)
    else:
        reference_spemb_for_plot = None
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batchsize,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8,
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=8,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    valid_loader = DataLoader(batch_size=10,
                              dataset=valid_dataset,
                              drop_last=False,
                              num_workers=5,
                              pin_memory=False,
                              prefetch_factor=2,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)

    loss_plot = [[], []]
    with open(os.path.join(save_directory, "config.txt"), "w+") as conf:
        conf.write(config)
    step_counter = 0
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)

    start_time = time.time()
    for epoch in range(epochs):
        # train one epoch
        grad_accum = 0
        optimizer.zero_grad()
        train_losses_this_epoch = list()
        for train_datapoint in train_loader:
            with autocast():
                if not spemb:
                    train_loss = net(train_datapoint[0].to(device),
                                     train_datapoint[1].to(device),
                                     train_datapoint[2].to(device),
                                     train_datapoint[3].to(device),
                                     train_datapoint[4].to(device),
                                     train_datapoint[5].to(device),
                                     train_datapoint[6].to(device))
                else:
                    train_loss = net(train_datapoint[0].to(device),
                                     train_datapoint[1].to(device),
                                     train_datapoint[2].to(device),
                                     train_datapoint[3].to(device),
                                     train_datapoint[4].to(device),
                                     train_datapoint[5].to(device),
                                     train_datapoint[6].to(device),
                                     train_datapoint[7].to(device))
                train_losses_this_epoch.append(float(train_loss))
            scaler.scale((train_loss / gradient_accumulation)).backward()
            del train_loss
            grad_accum += 1
            if grad_accum % gradient_accumulation == 0:
                grad_accum = 0
                step_counter += 1
                # update weights
                # print("Step: {}".format(step_counter))
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
        # evaluate on valid after every epoch is through
        with torch.no_grad():
            net.eval()
            val_losses = list()
            for validation_datapoint in valid_loader:
                if not spemb:
                    val_losses.append(float(net(validation_datapoint[0].to(device),
                                                validation_datapoint[1].to(device),
                                                validation_datapoint[2].to(device),
                                                validation_datapoint[3].to(device),
                                                validation_datapoint[4].to(device),
                                                validation_datapoint[5].to(device),
                                                validation_datapoint[6].to(device))))
                else:
                    val_losses.append(float(net(validation_datapoint[0].to(device),
                                                validation_datapoint[1].to(device),
                                                validation_datapoint[2].to(device),
                                                validation_datapoint[3].to(device),
                                                validation_datapoint[4].to(device),
                                                validation_datapoint[5].to(device),
                                                validation_datapoint[6].to(device),
                                                validation_datapoint[7].to(device))))
            average_val_loss = sum(val_losses) / len(val_losses)
            if epoch & epochs_per_save == 0:
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "step_counter": step_counter,
                            "scheduler": scheduler.state_dict()},
                           os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
                plot_progress_spec(net, device, save_dir=save_directory, step=step_counter, lang=lang,
                                   reference_spemb_for_plot=reference_spemb_for_plot)
            print("Epoch:        {}".format(epoch + 1))
            print("Train Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
            print("Valid Loss:   {}".format(average_val_loss))
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
            print("Steps:        {}".format(step_counter))
            loss_plot[0].append(sum(train_losses_this_epoch) / len(train_losses_this_epoch))
            loss_plot[1].append(average_val_loss)
            with open(os.path.join(save_directory, "train_val_loss.json"), 'w') as plotting_data_file:
                json.dump(loss_plot, plotting_data_file)
            net.train()
