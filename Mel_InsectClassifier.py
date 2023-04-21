from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
from torchaudio import transforms
import torchaudio
from torch import nn
from torch.nn import init
import os
import glob
import ntpath
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sn
import pandas as pd
import warnings
from pathlib import Path

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

# Model adapted from:
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

# specifications for operation
frontend = "MEL"                        # define frontend for training or evaluation
train_model = True                      # activate training
test_model = True                       # activate testing
test_point = 57                         # for testing trained model
early_stop = True                       # activate early stopping

# parameters
max_patience = 8                        # define ES patience
num_epochs = 4                         # number of epochs for training when ES is disabled
N_CHANNELS = 1                          # number of input channels
SAMPLE_RATE = 44100                     # sample rate of input files
audio_len = 5                           # length of audio files
HOP_LENGTH = int((SAMPLE_RATE*5)/1500)  # hop length for Mel
BATCH_SIZE = 14                         # batch size for training and evaluation
N_MELS = 64                             # number of filters for Mel
top_db = 80                             # set max dB for decibel conversion
seed = 100                              # set seed to control randomization

torch.manual_seed(seed)

if early_stop:                          # high number of epochs when ES is enabled
    num_epochs = 200

train_data_path = f"{os.getcwd()}/Dataset/Train Chunks Aug/"                 # Augmented training dataset
validation_data_path = f"{os.getcwd()}/Dataset/Validation Chunks/"    # Non-augmented validation dataset
test_data_path = f"{os.getcwd()}/Dataset/Test Chunks/"                # Non-augmented test dataset

Path(f"{os.getcwd()}/Results/").mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {frontend} on {device}")

# List of species and their ID
ID_list = {"Azanicadazuluensis":               0,
           "Brevisianabrevis":                 1,
           "Chorthippusbiguttulus":            2,
           "Chorthippusbrunneus":              3,
           "Grylluscampestris":                4,
           "Kikihiamuta":                      5,
           "Myopsaltaleona":                   6,
           "Myopsaltalongicauda":              7,
           "Myopsaltamackinlayi":              8,
           "Myopsaltamelanobasis":             9,
           "Myopsaltaxerograsidia":            10,
           "Nemobiussylvestris":               11,
           "Oecanthuspellucens":               12,
           "Pholidopteragriseoaptera":         13,
           "Platypleuracapensis":              14,
           "Platypleuracfcatenata":            15,
           "Platypleurachalybaea":             16,
           "Platypleuradeusta":                17,
           "Platypleuradivisa":                18,
           "Platypleurahaglundi":              19,
           "Platypleurahirtipennis":           20,
           "Platypleuraintercapedinis":        21,
           "Platypleuraplumosa":               22,
           "Platypleurasp04":                  23,
           "Platypleurasp10":                  24,
           "Platypleurasp11cfhirtipennis":     25,
           "Platypleurasp12cfhirtipennis":     26,
           "Platypleurasp13":                  27,
           "Pseudochorthippusparallelus":      28,
           "Pycnasemiclara":                   29,
           "Roeselianaroeselii":               30,
           "Tettigoniaviridissima":            31}


# creates dataframes with file path and class ID for all files in the three datasets
def annotate(data_path):
    species_list = []  # init augmented species list
    files_list = []  # init augmented files list
    classID = []  # init list of classIDs per species

    for filename in glob.glob(os.path.join(data_path, '*.wav')):
        file = ntpath.basename(filename)        # filename without path
        species = file.split("_")[0]            # only get characters containing the species
        ID = dict.get(ID_list, str(species))    # get species ID
        classID.append(ID)                      # get classIDs for each file
        species_list.append(species)            # append species and files to lists
        files_list.append(file)

    # combine lists, sort and create dataframe for data loader
    metadata_file = list(zip(classID, species_list, files_list))
    metadata_file.sort()
    col_names = ["classID", "species_name", "file_name"]
    df = pd.DataFrame(metadata_file, columns=col_names)
    df["path_name"] = data_path + df["file_name"]
    df = df[["path_name", "classID"]]
    return df


# unify signal sizes
def pad_truncate(sig, sr, dim):     #dim=1 for MEL

    if sig.size(dim=dim) < sr * audio_len:  # pad signal if too short
        pad_value = (sig.size(dim=dim) - sr * audio_len) * (-1)
        sig = F.pad(sig, (0, pad_value))

    if sig.size(dim=dim) > sr * audio_len:  # truncate signal if too long
        sig = sig.numpy()
        sig = sig[:sr * audio_len]
        sig = torch.tensor(sig)

    return sig


# open files, truncate number of samples, preprocess for frontends
class AudioUtil:
    @staticmethod
    def open(audio_file):                   # open files
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def spectro_gram(aud, n_mels=N_MELS):   # create spectrogram
        sig, sr = aud
        sig = pad_truncate(sig, sr, dim=1)

        spec = transforms.MelSpectrogram(SAMPLE_RATE, n_mels=n_mels, hop_length=HOP_LENGTH, n_fft=1000,
                                         f_max=SAMPLE_RATE/2, win_length=int(HOP_LENGTH*2))(sig)

        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)  # Convert to decibels

        return spec


# define datasets
class SoundDS:
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = data_path
        self.sr = SAMPLE_RATE
        self.channel = N_CHANNELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.df.loc[idx, "path_name"]      # Absolute file path of the audio file
        class_id = self.df.loc[idx, 'classID']          # Get the Class ID
        aud = AudioUtil.open(audio_file)                # open file

        if frontend == "MEL":                           # process file for Mel
            rep = AudioUtil.spectro_gram(aud)

        return rep, class_id


# create dataframes for datasets
train_df = annotate(train_data_path)
validation_df = annotate(validation_data_path)
test_df = annotate(test_data_path)

# create datasets
train_dataset = SoundDS(train_df, train_data_path)
validation_dataset = SoundDS(validation_df, validation_data_path)
test_dataset = SoundDS(test_df, test_data_path)

num_train = len(train_dataset)
num_val = len(validation_dataset)
num_test = len(test_dataset)
print(f"{num_train} training files with seed {seed} for {num_epochs} epochs with batch size {BATCH_SIZE}")

# Create training, validation and test data loaders
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# define model
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(N_CHANNELS, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(N_CHANNELS*8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(N_CHANNELS*16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(N_CHANNELS*32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=32)
        self.dropout = nn.Dropout(0.4)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)
        x = self.dropout(x)

        return x


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
myModel = myModel.to(device)


# training and validation
def training(model, train_dl, num_epochs):
    global checkpoint
    checkpoint = 0
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs, anneal_strategy='linear')
    # performance tracking parameters
    train_losses = []
    train_scores = []
    val_scores = []
    val_losses = []

    # early stop parameters
    last_loss = 0
    patience = 0

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        model.train()
        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            torch.cuda.empty_cache()
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        avg_train_loss = running_loss / len(train_dl)
        train_losses.append(avg_train_loss)
        train_acc = correct_prediction / total_prediction
        train_scores.append(train_acc)
        print(f'E{epoch+1} Training Loss: {avg_train_loss:.2f}, Accuracy: {train_acc:.2f}')

        # Disable gradient updates for validation
        with torch.no_grad():
            model.eval()
            correct_val_prediction = 0
            total_val_prediction = 0
            running_val_loss = 0.0
            for data in validation_dl:
                torch.cuda.empty_cache()
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Get predictions
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                # Keep stats for Loss and Accuracy
                running_val_loss += val_loss.item()

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)

                # Count of predictions that matched the target label
                correct_val_prediction += (prediction == labels).sum().item()
                total_val_prediction += prediction.shape[0]

        # Print stats at the end of the validation step
        val_acc = correct_val_prediction / total_val_prediction
        val_scores.append(val_acc)
        avg_val_loss = running_val_loss / len(validation_dl)
        val_losses.append(avg_val_loss)

        if not early_stop:
            print(f"E{epoch + 1} Validation Loss: {avg_val_loss:.2f}, Accuracy: {val_acc:.2f}")

        # Early stopping
        if early_stop:
            if epoch == 0:                      # first epoch
                last_loss = avg_val_loss
                checkpoint = epoch + 1          # save model
                torch.save(model, f"{os.getcwd()}/Results/{frontend}_{seed}_{checkpoint}.pth")
                print(f"E{epoch + 1} Validation Loss: {avg_val_loss:.2f}, Accuracy: {val_acc:.2f}, "
                      f"Patience: {patience}/{max_patience}")

            if epoch > 0:                                               # after first epoch
                if avg_val_loss <= last_loss:                           # if loss improved
                    patience = 0                                        # reset patience
                    last_loss = avg_val_loss                            # update ideal loss
                    os.remove(f"{os.getcwd()}/Results/{frontend}_{seed}_{checkpoint}.pth")
                    checkpoint = epoch + 1                              # delete old model state, save new model
                    torch.save(model, f"{os.getcwd()}/Results/{frontend}_{seed}_{checkpoint}.pth")
                    print(f"E{epoch + 1} Validation Loss: {avg_val_loss:.2f}, Accuracy: {val_acc:.2f}, "
                          f"Patience: {patience}/{max_patience}")

                if avg_val_loss > last_loss:                            # if loss does not improve
                    patience += 1                                       # update patience
                    print(f"E{epoch + 1} Validation Loss: {avg_val_loss:.2f}, Accuracy: {val_acc:.2f}, "
                          f"Patience: {patience}/{max_patience}")

                    if patience == max_patience:                        # if maximum patience is reached
                        print("Stop Training")

                        # plot model training performance
                        fig, axs = plt.subplots(2, 1, sharex="all")
                        axs[0].plot(train_scores, label="Train")
                        axs[0].plot(val_scores, label="Validation")
                        axs[0].legend(loc="lower right")
                        axs[0].set_ylabel("Accuracy")
                        axs[1].plot(train_losses)
                        axs[1].plot(val_losses)
                        axs[1].set_xlabel("Epoch")
                        axs[1].set_ylabel("Loss")
                        fig.suptitle(f"{frontend}_{SAMPLE_RATE}_{checkpoint}")
                        fig.subplots_adjust(hspace=.001)
                        axs[0].set_xticklabels(())
                        plt.xticks(np.arange(len(train_scores)), np.arange(1, len(train_scores) + 1), rotation=90)
                        plt.axvline(x=checkpoint-1, linestyle="dashed", color="r")
                        axs[0].title.set_visible(False)
                        plt.savefig(f"{os.getcwd()}/Results/Perf_{frontend}_{seed}_{checkpoint}.pdf")

                        break

    if not early_stop:
        # plot model training performance
        fig, axs = plt.subplots(2, 1, sharex="all")
        axs[0].plot(train_scores, label="Train")
        axs[0].plot(val_scores, label="Validation")
        axs[0].legend(loc="lower right")
        axs[0].set_ylabel("Accuracy")
        axs[1].plot(train_losses)
        axs[1].plot(val_losses)
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        fig.suptitle(f"{frontend}_{seed}_{checkpoint}")
        fig.subplots_adjust(hspace=.001)
        axs[0].set_xticklabels(())
        plt.xticks(np.arange(len(train_scores)), np.arange(1, len(train_scores) + 1), rotation=90)
        axs[0].title.set_visible(False)
        plt.savefig(f"{os.getcwd()}/Results/Perf_{frontend}_{seed}_{checkpoint}.pdf")

    return checkpoint


# test evaluation
def inference(model, test_dl):
    correct_test_prediction = 0
    total_test_prediction = 0
    model.eval()

    # for confusion matrix
    y_pred = []
    y_true = []

    # Disable gradient updates
    with torch.no_grad():
        counter = 0
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # record all true labels
            if counter == 0:
                label_list = labels
            else:
                label_list = torch.cat((label_list, labels), dim=0)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # record all predicted labels
            if counter == 0:
                prediction_list = prediction
                counter += 1
            else:
                prediction_list = torch.cat((prediction_list, prediction), dim=0)

            # Count of predictions that matched the target label
            correct_test_prediction += (prediction == labels).sum().item()
            total_test_prediction += prediction.shape[0]

            # for confusion matrix
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            cm_label = labels.data.cpu().numpy()
            y_true.extend(cm_label)  # Save Truth

    # calculate accuracy, f1 score, precision and recall
    test_acc = correct_test_prediction / total_test_prediction
    f1 = f1_score(label_list.cpu(), prediction_list.cpu(), average="macro")
    precision = precision_score(label_list.cpu(), prediction_list.cpu(), average="macro", zero_division=0)
    recall = recall_score(label_list.cpu(), prediction_list.cpu(), average="macro")

    print(f'Test Accuracy: {test_acc:.2f}, F1 score: {f1:.2f}, '
          f'Recall: {recall:.2f}, Precision: {precision:.2f}')

    # Build confusion matrix
    classes = ID_list
    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(bottom=0.19, left=0.19)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    sn.heatmap(df_cm, annot=True, fmt=".2f", cbar=False, square=True, linewidths=0.2, linecolor="dimgrey")
    plt.savefig(f"{os.getcwd()}/Results/CM_{frontend}_{seed}_{checkpoint}.pdf")


# train model if specified
if train_model:
    training(myModel, train_dl, num_epochs)
    # save fully trained model if early stopping is deactivated
    if not early_stop:
        torch.save(myModel, f"{os.getcwd()}/Results/{frontend}_{seed}_{checkpoint}.pth")

if not train_model:
    if test_model:
        checkpoint = test_point         # specify for testing saved model

ModelTest = f"/Results/{frontend}_{seed}_{checkpoint}.pth"

# load and test saved model if specified
if test_model:
    myModel = torch.load(os.getcwd() + ModelTest, map_location=torch.device(device))
    inference(myModel, test_dl)
