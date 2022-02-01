import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from losses.HistogramLoss import HistogramLoss

class Encoder(nn.Module):
    def __init__(self, hidden_dims, in_channels=3, encoded_dim=256):
        super().__init__()

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        # self.lin1 = nn.Linear(hidden_dims[-1]*4, 128)
        # self.lin2 = nn.Linear(128, encoded_dim)
        self.lin1 = nn.Linear(hidden_dims[-1]*4, encoded_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        x = self.lin1(x)
        x = self.relu(x)
        # x = self.lin2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dims, encoded_dim=256):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # self.lin1 = nn.Linear(encoded_dim, 128)
        # self.lin2 = nn.Linear(128, hidden_dims[-1]*4)
        self.lin1 = nn.Linear(encoded_dim, hidden_dims[-1]*4)

        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        # x = self.lin2(x)
        # x = self.relu(x)

        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.output_layer(x)

        return x

#PyTorch defined model
class CoupledAutoEncoderTransformer(nn.Module):
    """basenet for fer2013"""
    def __init__(self, encoded_dim=256):
        super(CoupledAutoEncoderTransformer, self).__init__()

        hidden_dims = [128, 256, 512]

        self.source_encoder = Encoder(hidden_dims, 3, encoded_dim)
        self.source_decoder = Decoder(hidden_dims, encoded_dim)

        hidden_dims.reverse()
        self.target_encoder = Encoder(hidden_dims, 3, encoded_dim)
        self.target_decoder = Decoder(hidden_dims, encoded_dim)

        self.transformer = nn.Transformer(d_model=encoded_dim)

    def forward(self, x_src, x_trg):
        x_src = self.source_encoder.forward(x_src)
        x_src_latent = x_src
        x_src = self.source_decoder.forward(x_src)

        x_trg = self.target_encoder.forward(x_trg)
        x_trg_latent = x_trg
        x_trg = self.target_decoder.forward(x_trg)

        x_trans = torch.unsqueeze(x_src_latent, 1)
        x_trg_latent = torch.unsqueeze(x_trg_latent, 1)
        x_trans = self.transformer(x_trans, x_trg_latent)
        x_trans = torch.squeeze(x_trans, 1)
        x_out_src = self.source_decoder.forward(x_trans)
        x_out = self.target_decoder.forward(x_trans)

        return x_src, x_trg, x_out, x_out_src

#The abstract model class, uses above defined class and is used in the train script
class CoupledAutoEncoderTransformermodel(BaseModel):
    """basenet for fer2013"""

    def __init__(self, configuration):
        super().__init__(configuration)

        self.gamma = configuration['gamma']

        #Initialize model defined above
        self.model = CoupledAutoEncoderTransformer(configuration['encoded_dim'])
        self.model.cuda()
        # self.models = [AutoEncoder(configuration['encoded_dim']).cuda() for i in range(4)]

        #Define loss function
        self.criterion_loss = HistogramLoss("emd", 256).cuda()
        self.secondary_loss = nn.MSELoss().cuda()
        #Define optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=configuration['lr'],
            betas=(configuration['momentum'], 0.999),
            weight_decay=configuration['weight_decay']
        )
        # self.optimizers = [torch.optim.Adam(self.models[i].parameters(),
        #                     lr=configuration['lr'],
        #                     betas=(configuration['momentum'], 0.999),
        #                     weight_decay=configuration['weight_decay']) for i in range(4)]

        #Need to include these arrays with the optimizers and names of loss functions and models
        #Will be used by other functions for saving/loading
        # self.optimizers = [self.optimizers[i] for i in range(4)]
        self.optimizers = [self.optimizer]
        self.loss_names = ['total']
        self.network_names = ['model']

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

    #Calls the models forwards function
    def forward(self):
        x_src = self.source
        x_trg = self.target
        self.output_src, self.output_trg, self.output_da, self.output_reg = self.model.forward(x_src, x_trg)
        return self.output_src, self.output_trg, self.output_da, self.output_reg

    #Computes the loss with the specified name (in this case 'total')
    def compute_loss(self):
        hist_loss = self.criterion_loss(self.output_da, self.target)
        secondary_loss_src = self.secondary_loss(self.output_src, self.source)
        secondary_loss_trg = self.secondary_loss(self.output_trg, self.target)
        secondary_loss_reg = self.secondary_loss(self.output_reg, self.source)
        self.loss_total = hist_loss[0] + hist_loss[1] + (0.5*secondary_loss_src + 0.5*secondary_loss_trg) + secondary_loss_reg

    #Compute backpropogation for the model
    def optimize_parameters(self):
        self.loss_total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

    #Test function for the model
    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        self.val_images.append(self.source)
        self.val_predictions.append(self.output_src)
        self.val_labels.append(self.source)

    #Should be run after each epoch, outputs accuracy
    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['Accuracy'] = val_accuracy

        if (visualizer != None):
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []


if __name__ == "__main__":
    net = TEMPLATEmodel().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
