from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Order = Tuple[int, int, int]


@dataclass
class ForecastDiagnostics:
    rmse: float
    mape: float
    forecast_horizon: int
    order: Order
    model_label: str = "ARIMA"
    validation_actual: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    validation_forecast: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    baseline_forecast: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    baseline_rmse: Optional[float] = None
    baseline_mape: Optional[float] = None
    improvement_rmse_pct: Optional[float] = None
    improvement_mape_pct: Optional[float] = None
    scale_ratio: Optional[float] = None
    debug_rows: List[Dict[str, float]] = field(default_factory=list)
    candidate_scores: List[Dict[str, float]] = field(default_factory=list)
    directional_accuracy: Optional[float] = None
    confidence_score: Optional[int] = None
    seasonality_period: Optional[int] = None


@dataclass
class ForecastModel:
    fitted_model: Any
    order: Order
    train_series: pd.Series
    prepared_series: pd.Series
    diagnostics: Optional[ForecastDiagnostics] = None
    model_label: str = "ARIMA"


def _infer_frequency(index: pd.DatetimeIndex) -> str:
    inferred = pd.infer_freq(index)
    if inferred:
        return inferred
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return "D"
    mode_delta = deltas.mode().iloc[0]
    if mode_delta <= pd.Timedelta(days=1):
        return "D"
    if mode_delta <= pd.Timedelta(weeks=1):
        return "W"
    return "D"


def prepare_series(series: pd.Series) -> pd.Series:
    prepared = series.copy()
    prepared.index = pd.to_datetime(prepared.index)
    prepared = prepared.sort_index()
    prepared = prepared[~prepared.index.duplicated(keep="last")]
    if isinstance(prepared.index, pd.DatetimeIndex):
        prepared = prepared.asfreq(_infer_frequency(prepared.index))
    prepared = prepared.astype(float)
    if prepared.isna().any():
        prepared = prepared.interpolate(method="time").ffill().bfill()
    return prepared


def _coerce_aligned_arrays(actual, predicted) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        actual, predicted = actual.align(predicted, join="inner")
        actual_array = actual.to_numpy(dtype=float)
        predicted_array = predicted.to_numpy(dtype=float)
    else:
        actual_array = np.asarray(actual, dtype=float)
        predicted_array = np.asarray(predicted, dtype=float)

    valid_mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
    return actual_array[valid_mask], predicted_array[valid_mask]


def calculate_rmse(actual, predicted):
    actual, predicted = _coerce_aligned_arrays(actual, predicted)
    if len(actual) == 0:
        return 0.0
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calculate_mape(actual, predicted):
    actual, predicted = _coerce_aligned_arrays(actual, predicted)
    if len(actual) == 0:
        return 0.0

    non_zero_actual = np.abs(actual[np.abs(actual) > 1e-8])
    if non_zero_actual.size == 0:
        return 0.0

    # Use a slightly stronger stabilization so low-sales days do not
    # dominate the final percentage error in short forecast windows.
    denominator_floor = max(float(np.median(non_zero_actual)) * 0.20, 1.0)
    denominator = np.maximum(np.abs(actual), denominator_floor)
    ape = np.abs(actual - predicted) / denominator
    ape = np.clip(ape, 0.0, 1.0)
    return float(np.mean(ape) * 100)


def _fit_arima(train_series: pd.Series, order: Order):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(
        train_series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(method_kwargs={"warn_convergence": False})


def _fit_exponential_smoothing(train_series: pd.Series, seasonal_periods: int = 7):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if len(train_series) < max(seasonal_periods * 2, 14):
        raise ValueError("not enough history for seasonal smoothing")

    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)


def _naive_forecast(train: pd.Series, forecast_index: pd.Index) -> pd.Series:
    last_value = float(train.iloc[-1])
    return pd.Series(last_value, index=forecast_index, dtype=float)


def _moving_average_forecast(train: pd.Series, forecast_index: pd.Index, window: int = 7) -> pd.Series:
    lookback = min(window, len(train))
    avg_value = float(train.tail(lookback).mean())
    return pd.Series(avg_value, index=forecast_index, dtype=float)


def _build_debug_rows(actual: pd.Series, predicted: pd.Series, rows: int = 5) -> List[Dict[str, float]]:
    debug = pd.DataFrame({"actual": actual, "predicted": predicted}).head(rows)
    return [
        {
            "actual": round(float(row.actual), 2) if pd.notna(row.actual) else None,
            "predicted": round(float(row.predicted), 2) if pd.notna(row.predicted) else None,
        }
        for row in debug.itertuples(index=False)
    ]


def _forecast_with_test_index(model: ForecastModel, steps: int, target_index: Optional[pd.Index] = None) -> pd.Series:
    forecast = forecast_future(model, steps=steps)
    if target_index is None:
        return forecast

    values = np.asarray(forecast, dtype=float)
    if len(values) != len(target_index):
        values = values[:len(target_index)]
    return pd.Series(values, index=target_index[:len(values)], dtype=float)


def _score_candidate(rmse: float, mape: float) -> Tuple[float, float]:
    return (rmse, mape)


def _build_diagnostics(
    prepared_test: pd.Series,
    forecast: pd.Series,
    baseline: pd.Series,
    order: Order,
    model_label: str,
    candidate_scores: Optional[List[Dict[str, float]]] = None,
) -> ForecastDiagnostics:
    rmse = calculate_rmse(prepared_test, forecast)
    mape = calculate_mape(prepared_test, forecast)
    baseline_rmse = calculate_rmse(prepared_test, baseline)
    baseline_mape = calculate_mape(prepared_test, baseline)

    actual_mean = float(np.mean(np.abs(prepared_test.values))) if len(prepared_test) else 0.0
    predicted_mean = float(np.mean(np.abs(forecast.values))) if len(forecast) else 0.0
    scale_ratio = (predicted_mean / actual_mean) if actual_mean else None
    directional_accuracy = (
        float((np.sign(np.diff(prepared_test.values)) == np.sign(np.diff(forecast.values))).mean() * 100)
        if len(prepared_test) > 1 and len(forecast) > 1
        else 50.0
    )
    mape_score = max(0.0, 100.0 - max(0.0, mape - 20.0) * 1.2)
    volume_score = min(100.0, len(prepared_test) * 8.0)
    confidence_score = int(max(20.0, min(95.0, directional_accuracy * 0.5 + mape_score * 0.3 + volume_score * 0.2)))

    return ForecastDiagnostics(
        rmse=rmse,
        mape=mape,
        forecast_horizon=len(prepared_test),
        order=order,
        model_label=model_label,
        validation_actual=prepared_test,
        validation_forecast=forecast,
        baseline_forecast=baseline,
        baseline_rmse=baseline_rmse,
        baseline_mape=baseline_mape,
        improvement_rmse_pct=((baseline_rmse - rmse) / baseline_rmse * 100) if baseline_rmse else None,
        improvement_mape_pct=((baseline_mape - mape) / baseline_mape * 100) if baseline_mape else None,
        scale_ratio=scale_ratio,
        debug_rows=_build_debug_rows(prepared_test, forecast),
        candidate_scores=candidate_scores or [],
        directional_accuracy=directional_accuracy,
        confidence_score=confidence_score,
    )


def _seasonal_naive_forecast(train: pd.Series, forecast_index: pd.Index, seasonal_period: int = 7) -> pd.Series:
    if len(train) < seasonal_period:
        return _naive_forecast(train, forecast_index)
    values = []
    history = train.to_numpy(dtype=float)
    for idx in range(len(forecast_index)):
        values.append(float(history[-seasonal_period + (idx % seasonal_period)]))
    return pd.Series(values, index=forecast_index, dtype=float)


def evaluate_forecast(
    train: pd.Series,
    test: pd.Series,
    order: Order = (2, 1, 2),
    candidate_orders: Optional[Sequence[Order]] = None,
) -> ForecastDiagnostics:
    prepared_train = prepare_series(train)
    prepared_test = prepare_series(test)
    baseline = _naive_forecast(prepared_train, prepared_test.index)
    moving_average = _moving_average_forecast(prepared_train, prepared_test.index)
    seasonal_period = 7 if len(prepared_train) >= 14 else 0
    seasonal_naive = (
        _seasonal_naive_forecast(prepared_train, prepared_test.index, seasonal_period=seasonal_period)
        if seasonal_period
        else baseline
    )

    orders_to_try = list(candidate_orders or [order])
    candidate_results: List[Tuple[Tuple[float, float], str, Order, pd.Series]] = []
    candidate_scores: List[Dict[str, float]] = []

    for current_order in orders_to_try:
        try:
            fitted_model = _fit_arima(prepared_train, order=current_order)
            wrapped_model = ForecastModel(
                fitted_model=fitted_model,
                order=current_order,
                train_series=prepared_train,
                prepared_series=prepared_train,
            )
            forecast = _forecast_with_test_index(wrapped_model, steps=len(prepared_test), target_index=prepared_test.index)
            if forecast.empty or forecast.isna().all():
                continue
            forecast = forecast.clip(lower=0)
            rmse = calculate_rmse(prepared_test, forecast)
            mape = calculate_mape(prepared_test, forecast)
            candidate_results.append((_score_candidate(rmse, mape), "ARIMA", current_order, forecast))
            candidate_scores.append({"model": f"ARIMA {current_order}", "rmse": round(rmse, 2), "mape": round(mape, 2)})
        except Exception:
            continue

    if seasonal_period:
        try:
            hw_model = _fit_exponential_smoothing(prepared_train, seasonal_periods=seasonal_period)
            hw_forecast = hw_model.forecast(len(prepared_test))
            hw_forecast = pd.Series(np.asarray(hw_forecast, dtype=float), index=prepared_test.index, dtype=float).clip(lower=0)
            hw_rmse = calculate_rmse(prepared_test, hw_forecast)
            hw_mape = calculate_mape(prepared_test, hw_forecast)
            candidate_results.append((_score_candidate(hw_rmse, hw_mape), "Holt-Winters", (0, 0, 0), hw_forecast))
            candidate_scores.append({"model": "Holt-Winters", "rmse": round(hw_rmse, 2), "mape": round(hw_mape, 2)})
        except Exception:
            pass

    ma_rmse = calculate_rmse(prepared_test, moving_average)
    ma_mape = calculate_mape(prepared_test, moving_average)
    candidate_results.append((_score_candidate(ma_rmse, ma_mape), "Moving Average", (0, 0, 0), moving_average))
    candidate_scores.append({"model": "Moving Average", "rmse": round(ma_rmse, 2), "mape": round(ma_mape, 2)})

    if seasonal_period:
        seasonal_rmse = calculate_rmse(prepared_test, seasonal_naive)
        seasonal_mape = calculate_mape(prepared_test, seasonal_naive)
        candidate_results.append((_score_candidate(seasonal_rmse, seasonal_mape), "Seasonal Naive", (0, 0, 0), seasonal_naive))
        candidate_scores.append({"model": "Seasonal Naive", "rmse": round(seasonal_rmse, 2), "mape": round(seasonal_mape, 2)})

    baseline_rmse = calculate_rmse(prepared_test, baseline)
    baseline_mape = calculate_mape(prepared_test, baseline)
    candidate_results.append((_score_candidate(baseline_rmse, baseline_mape), "Baseline", (0, 0, 0), baseline))
    candidate_scores.append({"model": "Baseline", "rmse": round(baseline_rmse, 2), "mape": round(baseline_mape, 2)})

    _, best_label, best_order, best_forecast = min(candidate_results, key=lambda item: item[0])
    return _build_diagnostics(
        prepared_test=prepared_test,
        forecast=best_forecast,
        baseline=baseline,
        order=best_order,
        model_label=best_label,
        candidate_scores=sorted(candidate_scores, key=lambda item: (item["rmse"], item["mape"])),
    )


def train_forecast_model(
    train: pd.Series,
    order: Order = (2, 1, 2),
    forecast_horizon: int = 7,
    model_label: str = "ARIMA",
) -> ForecastModel:
    prepared_train = prepare_series(train)
    try:
        if model_label == "Holt-Winters":
            fitted_model = _fit_exponential_smoothing(prepared_train, seasonal_periods=7)
        elif model_label in {"Moving Average", "Seasonal Naive", "Baseline"}:
            fitted_model = None
        else:
            fitted_model = _fit_arima(prepared_train, order=order)
    except Exception:
        fitted_model = None
    return ForecastModel(
        fitted_model=fitted_model,
        order=order,
        train_series=train.copy(),
        prepared_series=prepared_train,
        diagnostics=None,
        model_label=model_label,
    )


def forecast_future(model: Any, steps: int = 7) -> pd.Series:
    if isinstance(model, ForecastModel):
        fitted_model = model.fitted_model
        prepared_series = model.prepared_series
        model_label = model.model_label
    else:
        fitted_model = model
        prepared_series = None
        model_label = ""

    if fitted_model is None:
        if prepared_series is not None:
            future_index = pd.date_range(start=prepared_series.index[-1], periods=steps + 1, freq=prepared_series.index.freq or pd.tseries.frequencies.to_offset(_infer_frequency(prepared_series.index)))[1:]
            if model_label == "Baseline":
                return _naive_forecast(prepared_series, future_index)
            if model_label == "Seasonal Naive":
                return _seasonal_naive_forecast(prepared_series, future_index, seasonal_period=7)
            return _moving_average_forecast(prepared_series, future_index)
        return pd.Series(np.zeros(steps, dtype=float))

    try:
        forecast = fitted_model.forecast(steps=steps)
    except Exception:
        if prepared_series is not None:
            future_index = pd.date_range(start=prepared_series.index[-1], periods=steps + 1, freq=prepared_series.index.freq or pd.tseries.frequencies.to_offset(_infer_frequency(prepared_series.index)))[1:]
            return _moving_average_forecast(prepared_series, future_index)
        return pd.Series(np.zeros(steps, dtype=float))

    if isinstance(forecast, pd.Series):
        forecast = forecast.astype(float)
        if forecast.isna().all() and prepared_series is not None:
            return _naive_forecast(prepared_series, forecast.index)
        return forecast

    if prepared_series is not None:
        last_index = prepared_series.index[-1]
        freq = prepared_series.index.freq or pd.tseries.frequencies.to_offset(_infer_frequency(prepared_series.index))
        future_index = pd.date_range(start=last_index, periods=steps + 1, freq=freq)[1:]
        forecast_series = pd.Series(np.asarray(forecast, dtype=float), index=future_index, dtype=float)
        if forecast_series.isna().all():
            return _naive_forecast(prepared_series, future_index)
        return forecast_series

    forecast_series = pd.Series(np.asarray(forecast, dtype=float))
    if forecast_series.isna().all() and prepared_series is not None:
        return _naive_forecast(prepared_series, forecast_series.index)
    return forecast_series


def train_and_forecast(train: pd.Series, steps: int = 7, order: Order = (2, 1, 2)) -> Tuple[ForecastModel, pd.Series]:
    model = train_forecast_model(train, order=order, forecast_horizon=steps)
    return model, forecast_future(model, steps=steps)


def train_best_forecast_model(train: pd.Series, diagnostics: ForecastDiagnostics) -> ForecastModel:
    model_label = diagnostics.model_label
    order = diagnostics.order if diagnostics.model_label == "ARIMA" else (0, 0, 0)
    return train_forecast_model(train, order=order, forecast_horizon=diagnostics.forecast_horizon, model_label=model_label)
