"""Public quant research primitives for microstructure and backtesting work."""

from .backtest import BacktestConfig, BacktestReport, run_backtest, walk_forward_splits
from .baselines import (
    LinearDirectionModel,
    EWMAVolatilityModel,
    fit_ridge_direction_model,
    predict_direction_score,
    predict_trade_action,
    predict_ewma_volatility,
)
from .core import FeatureFrame, LabelFrame, MarketEvent
from .features import (
    book_imbalance,
    depth_imbalance,
    realized_volatility,
    microprice,
    midprice,
    order_flow_imbalance,
    signed_volume,
)
from .labels import label_direction, triple_barrier_label
from .models import MLPDirectionArtifact, train_mlp_direction_model, build_prediction_rows
