from Dataset_new import Dataset
from datetime import datetime as dt
from Model2 import LSTMModel, get_model
from model3 import GRUModel
from model1 import Hybrid_GRU_LSTM_model, get_model


if __name__ == "__main__":
	stock = Dataset("HDFCBANK", start_date=dt(2019, 11, 1),  end_date=dt(2022, 3, 31))
	stock.plot_history()
	model = GRUModel(stock.X_train, stock.y_train)
	# model = LSTMModel(stock.X_train, stock.y_train)
	# model = Hybrid_GRU_LSTM_model(stock.X_train, stock.y_train)
	# model = get_model()
	model.train(epochs=15, batch_size=32)
	

	predicted_price_scaled, predicted_price = stock.predict()
	stock.compare_plot()
	metrics = stock.find_mse()
	print("MSE = ",metrics["MSE"])
	print("RMSE = ",metrics["RMSE"])
	print("MAE = ",metrics["MAE"])

