from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.parse_argument import parse_argument
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics
from homework.src._internals.save_model_if_better import save_model_if_better
from homework.src._internals.select_model import select_model

TEST_SIZE = 0.25
RANDOM_STATE = 123456


def main():
    args = parse_argument()
    model = select_model(args)

    x_train, x_test, y_train, y_test = prepare_data()

    model.fit(x_train, y_train)

    mse, mae, r2 = calculate_metrics(x_train, y_train, model)
    print_metrics(mse, mae, r2, title="Metricas de entrenamiento:")

    mse, mae, r2 = calculate_metrics(x_test, y_test, model)
    print_metrics(mse, mae, r2, title="Metricas de testing:")

    save_model_if_better(model, x_test, y_test)


if __name__ == "__main__":
    main()
