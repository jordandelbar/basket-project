"""Helper functions for training and evaluation."""
import torch
from loguru import logger


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.nn.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Args:
        model (torch.nn.Module): model to train
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        train_dataloader (torch.utils.data.DataLoader): training dataloader
        device (torch.device): device to use for training
    """
    model.train()
    train_loss = 0

    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # Backward pass
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_dataloader)


def validate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Validate the model for one epoch.

    Args:
        model (torch.nn.Module): model to validate
        criterion (torch.nn.Module): loss function
        val_dataloader (torch.utils.data.DataLoader): validation dataloader
        device (torch.device): device to use for validation
    """
    model.eval()
    val_loss = 0

    for x, y in val_dataloader:
        x_items = x.to(device)
        y = y.to(device)

        # Forward pass
        y_pred = model(x_items)
        loss = criterion(y_pred, y)
        val_loss += loss.item()

    return val_loss / len(val_dataloader)


def training_loop(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.nn.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
) -> tuple:
    """Training loop.

    Args:
        model (torch.nn.Module): model to train
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        train_dataloader (torch.utils.data.DataLoader): training dataloader
        val_dataloader (torch.utils.data.DataLoader): validation dataloader
        num_epochs (int): number of epochs to train
        device (torch.device): device to use for training
    """
    # Define necessary variables
    best_loss = float("inf")
    epochs_since_improvement = 0
    patience = 8  # Number of epochs to wait for improvement

    # Track metrics
    train_losses = []
    val_losses = []

    model = model.to(device)
    criterion = criterion  # .to(device)

    embedding_hist = []

    for params in model.parameters():
        params.requires_grad = True

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            device=device,
        )  # Perform training steps
        train_losses.append(train_loss)  # Track training loss

        # Calculate validation loss
        val_loss = validate(
            model=model,
            criterion=criterion,
            val_dataloader=val_dataloader,
            device=device,
        )  # Perform validation steps
        val_losses.append(val_loss)  # Track validation loss

        logger.info(
            (f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}- Validation Loss: {val_loss:.4f}")
        )

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_since_improvement += 1

        # Check if early stopping criteria met
        if epochs_since_improvement > patience:
            logger.info(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

        embedding_hist.append(model.embeddings.weight.data)

    # Load the best model checkpoint
    model.load_state_dict(torch.load("best_model.pt"))

    return model, train_losses, val_losses, embedding_hist
