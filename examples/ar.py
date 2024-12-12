import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
import torch

from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR


TRAINER_KWARGS = {"accelerator": "cpu"}


def fine_optimal_lr(trainer, model, train_dataloader, val_dataloader):
    model.hparams.log_interval = -1
    model.hparams.log_val_interval = -1
    trainer.limit_train_batches = 1.0
    res = Tuner(trainer).lr_find(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
    model.hparams.learning_rate = res.suggestion()
    return model.hparams.learning_rate


def fit_model(trainer, model, train_loader, val_dataloader):
    torch.set_num_threads(10)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_dataloader,
    )


# calcualte mean absolute error on validation set
def eval_predictions(model, val_dataloader, TRAINER_KWARGS):
    actuals = torch.cat([y for _, (y, _) in iter(val_dataloader)])
    predictions = model.predict(val_dataloader, trainer_kwargs=TRAINER_KWARGS)
    return (actuals - predictions).abs().mean()


def plot_prediction(model, val_dataloader):
    raw_predictions, x = model.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=TRAINER_KWARGS)
    for idx in range(10):  # plot 10 examples
        model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)


def main():
    data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100)
    data["static"] = "2"
    data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
    validation = data.series.sample(20)

    max_encoder_length = 60
    max_prediction_length = 20

    training = TimeSeriesDataSet(
        data[lambda x: ~x.series.isin(validation)],
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        static_categoricals=["static"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["value"],
        time_varying_known_reals=["time_idx"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=False,
        add_target_scales=True,
        randomize_length=None,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        data[lambda x: x.series.isin(validation)],
        stop_randomization=True,
    )
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        gradient_clip_val=0.1,
        limit_train_batches=30,
        limit_val_batches=3,
        callbacks=[lr_logger, early_stop_callback],
    )
    deepar = DeepAR.from_dataset(
        training,
        learning_rate=0.1,
        hidden_size=32,
        dropout=0.1,
        loss=NormalDistributionLoss(),
        log_interval=10,
        log_val_interval=3,
    )
    print(f"Number of parameters in network: {deepar.size() / 1e3:.1f}k")
    # fit model
    fit_model(trainer, deepar, train_dataloader, val_dataloader)
    # evaluate model
    mae = eval_predictions(deepar, val_dataloader, TRAINER_KWARGS)
    print(f"Mean absolute error of model: {mae}")


if __name__ == "__main__":
    main()
