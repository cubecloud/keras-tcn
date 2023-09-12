# https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras import Input, Model
from keras.layers import Dense, Embedding, Bidirectional
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from keras.losses import BinaryFocalCrossentropy
from sklearn.preprocessing import MinMaxScaler

from tcn import TCN

from tsmodels.moderncallbacks import *
from datawizard.keeper import Keeper
from datawizard.dataload import DataLoad
from datawizard.featurizer import Featurizer
from datawizard.tsgenerators import TSRollingGenerator, ExperimentalRollingGenerator
from datawizard.horizons import HorizonsDataLoader

##
# It's a very naive (toy) example to show how to do time series forecasting.
# - There are no training-testing sets here. Everything is training set for simplicity.
# - There is no input/output normalization.
# - The model is simple.
##

# milk = pd.read_csv('monthly-milk-production-pounds-p.csv', index_col=0, parse_dates=True)
#
# print(milk.head())
# lookback_window = 12  # months.
# milk = milk.values  # just keep np array here for simplicity.

symbol = 'BTCUSDT'
interval = '1h'
power_trend = 0.056
support_trend = 0.075
timeframes_period = 276
signal_direction = 'plus'
# signal_direction = 'minus'
target_steps = 1
freq = '7m'

pairs_data_path = os.path.join(Keeper.get_data_folder_path(), "pairs_data")
hdl = HorizonsDataLoader(pair_symbol=symbol,
                         source_directory=pairs_data_path,
                         use_period=('2019-10-01 00:00:00',
                                     '2023-12-25 00:00:00'
                                     ),
                         horizons_list=["1m", "1h"],
                         verbose=0,
                         )

freqs_list = hdl.timeframe_with_freq(timeframe="1h", freq=freq, origin="start")

features_dict = {
    "P_trend": [["Open", "High", "Low", "Close"], f"_trend_power({str(power_trend)})"],
    # "Signal": [["Open", "High", "Low", "Close"], f"_trend_power({str(power_trend)})", "_to_classes"],
    "[Signal_0,Signal_1]": ["P_trend", f"_buy_sell_markers(3)", f"_filtered_markers({signal_direction})",
                            "_onehotencoder"],
    # "NATR": [["Open", "High", "Low", "Close"], "-talib.NATR(14)"],
    "Ret_perc": ["Close", ".diff(1)", "Close", "div"],
    f"Ret_perc_{timeframes_period}": ["Close", f".diff({timeframes_period})", "Close", "div"],
    f"Ret_perc_{int(timeframes_period/2)}": ["Close", f".diff({int(timeframes_period/2)})", "Close", "div"],
    f"Ret_perc_{int(timeframes_period/4)}": ["Close", f".diff({int(timeframes_period/4)})", "Close", "div"],
    f"Ret_perc_{int(timeframes_period/8)}": ["Close", f".diff({int(timeframes_period/8)})", "Close", "div"],
    f"Ret_perc_24": ["Close", f".diff(24)", "Close", "div"],
    f"Ret_perc_{int(timeframes_period/16)}": ["Close", f".diff({int(timeframes_period/16)})", "Close", "div"],
    # "BOP": [["Open", "High", "Low", "Close"], "-talib.BOP"],
    # f"Vwap_{timeframes_period}": ["High", "Low", "Close", "add", "add", 3.0, "div", "Volume", "mul",
    #                               f".rolling({timeframes_period})", ".sum",
    #                               "Volume", f".rolling({timeframes_period})", ".sum",
    #                               "div"],
    f"Vwap_24": ["High", "Low", "Close", "add", "add", 3.0, "div", "Volume", "mul",
                 f".rolling(24)", ".sum",
                 "Volume", f".rolling(24)", ".sum",
                 "div"],
    # "[HT_SINE_sine,HT_SINE_leadsine]": [["Open", "High", "Low", "Close"], "-talib.HT_SINE"],
    # "Signal": ["P_trend", "_buy_sell_markers(3)", "_filtered_markers(minus)"],
    # "Mm_vlm": ["Volume", "_minmaxscaler"],
    # "[MACD,MACDsignal,MACDhist]": [["Open", "High", "Low", "Close"], "-talib.MACD(12,26,9)"],
    # "[Rb_macd,Rb_macdsignal,Rb_macdhist]": [["MACD", "MACDsignal", "MACDhist"], "_robustscaler"],
    # "[Std_macd,Std_macdsignal,Std_macdhist]": [["MACD", "MACDsignal", "MACDhist"], "_standardscaler"],
    # "[Mm_macd,Mm_macdsignal,Mm_macdhist]": [["MACD", "MACDsignal", "MACDhist"], "_minmaxscaler"],
}

remove_columns = [
    "Volume",
    "P_trend",
    # "MACD",
    # "MACDsignal",
    # "MACDhist",
]

remove_columns.extend(["Signal_0", "Signal_1"])

train_data_list = list()
train_targets_list = list()

val_data_list = list()
val_targets_list = list()

test_data_list = list()
test_targets_list = list()

for _df in freqs_list:
    ftz = Featurizer(_df,
                     pair_symbol=symbol,
                     timeframe=interval,
                     operations_dict=features_dict,
                     remove_columns_list=remove_columns,
                     verbose=0,
                     )
    features_df, _ = ftz.run()
    _data = features_df.drop(columns=remove_columns)
    print(_data[timeframes_period:timeframes_period + 10].to_string())
    # _data_list.append(_data)
    _targets = features_df[["Signal_0", "Signal_1"]]
    print(_targets[timeframes_period:timeframes_period + 10].to_string())
    # _targets_list.append(_targets)
    # print(features_df.head(10).to_string())

    train_data = _data[(_data.index < "2022-07-31 23:00:00")]
    train_targets = _targets[(_targets.index < "2022-07-31 23:00:00")]
    train_data_list.append(train_data)
    train_targets_list.append(train_targets)

    val_data = _data[(_data.index >= "2022-07-31 23:00:00") & (_data.index < "2022-09-30 23:00:00")]
    val_targets = _targets[(_targets.index >= "2022-07-31 23:00:00") & (_targets.index < "2022-09-30 23:00:00")]
    val_data_list.append(val_data)
    val_targets_list.append(val_targets)

    test_data = _data[(_data.index >= "2022-09-30 23:00:00")]
    test_targets = _targets[(_targets.index >= "2022-09-30 23:00:00")]
    test_data_list.append(test_data)
    test_targets_list.append(test_targets)

train_gen = ExperimentalRollingGenerator(data=train_data_list,
                                         targets=train_targets_list,
                                         length=timeframes_period,
                                         apply_func='standardnormalizer',
                                         apply_cols=[["Open",
                                                      "High",
                                                      "Low",
                                                      "Close",
                                                      # f"Vwap_{timeframes_period}",
                                                      "Vwap_24"]
                                                     # "Volume",
                                                     # "MACD",
                                                     # "MACDsignal",
                                                     # "MACDhist",
                                                     ],
                                         trend_calc=True,
                                         power_trend=power_trend,
                                         # support_trend=support_trend,
                                         batch_size=64,
                                         overlap=0,
                                         start_index=timeframes_period + 1,
                                         target_steps=target_steps
                                         )

val_gen = ExperimentalRollingGenerator(data=val_data_list,
                                       targets=val_targets_list,
                                       length=timeframes_period,
                                       apply_func='standardnormalizer',
                                       apply_cols=[["Open",
                                                    "High",
                                                    "Low",
                                                    "Close",
                                                    # f"Vwap_{timeframes_period}",
                                                    "Vwap_24"]
                                                   # "Volume",
                                                   # "MACD",
                                                   # "MACDsignal",
                                                   # "MACDhist",
                                                   ],
                                       trend_calc=True,
                                       power_trend=power_trend,
                                       # support_trend=support_trend,
                                       batch_size=64,
                                       overlap=0,
                                       start_index=0,
                                       target_steps=target_steps
                                       )

test_gen = TSRollingGenerator(data=test_data_list[0],
                              targets=test_targets_list[0],
                              targets_type="ohe",
                              length=timeframes_period,
                              apply_func="standardnormalizer",
                              apply_cols=[["Open",
                                           "High",
                                           "Low",
                                           "Close",
                                           # f"Vwap_{timeframes_period}",
                                           "Vwap_24"]
                                          # "Volume",
                                          # "MACD",
                                          # "MACDsignal",
                                          # "MACDhist",
                                          ],
                              trend_calc=True,
                              power_trend=power_trend,
                              # support_trend=support_trend,
                              batch_size=timeframes_period,
                              overlap=0,
                              start_index=0,
                              target_steps=target_steps
                              )

lookback_window = timeframes_period  # hours.

# noinspection PyArgumentEqualDefault
tcn_layer = TCN(input_shape=(lookback_window, 15),
                nb_filters=64,
                kernel_size=2,
                nb_stacks=1,
                dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                # dropout_rate=0.15,
                # activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                use_skip_connections=True,
                use_batch_norm=True,
                use_weight_norm=False,
                use_layer_norm=False
                )
print("Receptive field:", tcn_layer.receptive_field)

model = Sequential([tcn_layer,
                    Dense(units=2, activation='softmax')
                    ])

# inputs = Input(shape=(lookback_window, 9), dtype="int32")
# x = Bidirectional(TCN(kernel_size=2,
#                       dilations=[1, 2, 4, 8, 16, 32, 64],
#                       use_skip_connections=True,
#                       use_batch_norm=True,
#                       use_weight_norm=False,
#                       use_layer_norm=False, ),
#                   )(inputs)
# outputs = Dense(2, activation="softmax")(x)
# model = Model(inputs, outputs)

# tf.keras.utils.plot_model(
#     model,
#     to_file='TCN_timeseries_model.png',
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir='TB',
#     dpi=200,
#     layer_range=None,
# )

tensorboard = TensorBoard(
    log_dir='logs_tcn',
    histogram_freq=1,
    write_images=True
)

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# bfc_loss = BinaryFocalCrossentropy(alpha=0.023,
#                                    gamma=1.0)
#
# model.compile('adam', bfc_loss, metrics=['accuracy'])
epochs = 400
warmup = 50
learning_rate = 1e-6
path_filename = f"weights_{signal_direction}"
monitor = "val_accuracy"
monitor_mode = "auto"
verbose = True
es_patience = 35
rlrs_patience = 5

lrcosh = CoSheduller(warmup=warmup,
                     learning_rate=learning_rate,
                     min_learning_rate=1e-7,
                     total_epochs=epochs,
                     )

chkp = ModernModelCheckpoint(f"{path_filename}.h5",
                             monitor=monitor,
                             mode=monitor_mode,
                             warmup=warmup,
                             # save_freq='epoch',
                             save_best_only=True,
                             verbose=verbose,
                             )
rlrs = ReduceLROnPlateau(monitor=monitor,
                         factor=0.2,
                         patience=rlrs_patience,
                         min_lr=1e-07
                         )
es = ModernEarlyStopping(patience=es_patience,
                         monitor=monitor,
                         mode=monitor_mode,
                         restore_best_weights=True,
                         verbose=1,
                         warmup=warmup
                         )

lrs = LearningRateScheduler(lrcosh.scheduler,
                            verbose=verbose)

callbacks = [rlrs, chkp, es, lrs, tensorboard]

print('Train...')
model.fit(train_gen,
          validation_data=val_gen,
          epochs=epochs,
          verbose=2,
          callbacks=callbacks,
          class_weight=train_gen.classes_weights
          )

path_filename = f"weights_{signal_direction}.h5"
print(f'Load weights: {path_filename}')
model.load_weights(path_filename)

print('Test...')
predictions = model.predict(test_gen)

scaler = MinMaxScaler()

predictions[:, 1] = np.where(predictions[:, 1] > 0.94, 1.0, 0.0)

show_df = pd.DataFrame(test_data_list[0]["Close"].copy(deep=True), columns=["Close"])
show_df["Signal_1"] = test_targets_list[0]["Signal_1"]

# reshape the close values to a 2D array with shape (n_samples, n_features)
# vwap_reshaped = test_data_list[0][f"Vwap_{timeframes_period}"].values.reshape(-1, 1)
vwap_24_reshaped = test_data_list[0]["Vwap_24"].values.reshape(-1, 1)
close_reshaped = test_data_list[0]["Close"].values.reshape(-1, 1)

# fit the scaler on the close values and transform them
close_scaled = scaler.fit_transform(close_reshaped)
# vwap_scaled = scaler.transform(vwap_reshaped)
vwap_24_scaled = scaler.transform(vwap_24_reshaped)

show_df["Close"] = close_scaled
show_df["Predict"] = np.zeros(show_df.shape[0])
show_df["Predict"][timeframes_period + target_steps:] = predictions[:-target_steps, 1]
# show_df[f"Vwap_{timeframes_period}"] = vwap_scaled
show_df["Vwap_24"] = vwap_24_scaled

print('Plot data...')
pd.DataFrame(show_df).plot()
plt.title('Hourly BTCUSDT prediction)')
plt.legend()
plt.show()
