import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device):
        super(MLP, self).__init__()

        self.projection_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        ).to(device)

    def forward(self, x):
        return self.projection_layer(x)


class StudentWrapper(nn.Module):
    def __init__(self, student_net, projection_head, prediction_head):
        super(StudentWrapper, self).__init__()
        self.student_net = student_net
        self.projection_head = projection_head
        self.prediction_head = prediction_head

    def forward(self, x):
        x = self.student_net(x)
        x = self.projection_head(x)
        x = self.prediction_head(x)
        return x


class LinearEvaluator(nn.Module):
    def __init__(self, mock_input, n_classes, device, lr=1e-4, epochs=30):
        super(LinearEvaluator, self).__init__()
        in_dims = mock_input.flatten(start_dim=1).shape[1]

        self.flatten = nn.Flatten(start_dim=1)
        self.linear_evaluator = nn.Linear(in_dims, n_classes)
        self.optimizer = torch.optim.Adam(self.linear_evaluator.parameters(), lr)
        self.epochs = epochs

        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        self.to(self.device)

    def forward_backward(self, x, y, student_model, backward=True):
        student_out = student_model(x)
        _, loss, preds = self(student_out, y)
        if backward:
            self.backward(loss)
        return loss, preds

    def forward(self, x, y):
        x = self.flatten(x)
        x = self.linear_evaluator(x)

        loss = self.loss_fn(x, y)
        predictions = torch.argmax(x, dim=1)
        return x, loss, predictions

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
