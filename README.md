# BTC Orderflow Backtester

Motor de backtest para **BTCUSDT perpetuo en velas de 5 minutos**, basado en flujo de órdenes
(delta, CVD, absorción) y posicionamiento (open interest, funding, ratios long/short). Combina
seis señales en un sistema de **confluencia ponderada** y evalúa si la estrategia superaría
el challenge de una **prop firm** (objetivo de beneficio, drawdown máximo y pérdida diaria).

El foco está en la ingeniería del backtest: datos multi-exchange alineados, cero lookahead
verificado automáticamente, validación walk-forward y Monte Carlo.

> Proyecto personal de investigación. Nada de esto es consejo de inversión.

## Arquitectura

```
config/            settings.yaml (parámetros) · prop_firm_rules.yaml (reglas de 3 perfiles de prop firm)
scripts/           descarga histórica · ejecución del backtest · diagnóstico de señales y de violaciones
src/
  data/            clientes de Binance, Bybit y Coinalyze · limpieza y alineación · almacenamiento Parquet
  features/        OHLCV · volatilidad · orderflow · open interest · funding · sentimiento · sesiones
  signals/         interfaz BaseSignal · 6 señales · motor de confluencia
  execution/       entrada y salida · tamaño de posición · límites diarios
  backtest/        motor · cartera · métricas · walk-forward · Monte Carlo · detector de lookahead
```

Cada capa depende solo de la anterior: datos → features → señales → ejecución → backtest.
Todos los parámetros salen de YAML, sin valores fijos en el código.

## Datos

- **Binance Futures:** velas OHLCV, funding, volumen taker buy/sell y **aggTrades** (cada
  operación ejecutada) descargados de `data.binance.vision`, agregados a delta por vela de 5 min.
- **Bybit V5:** open interest y ratio long/short en 5 min. Bybit conserva el histórico desde el
  lanzamiento del par, mientras que Binance solo guarda 30 días.
- **Coinalyze:** open interest agregado de todos los exchanges, como contexto.

Un pipeline de limpieza valida huecos, rellena los pequeños con un límite de velas, alinea todas
las fuentes a una rejilla exacta de 5 min y las une en un único DataFrame.

## Señales

| Señal | Lógica |
|---|---|
| **Liquidation sweep** | El precio barre una zona con mucho open interest, el OI cae de golpe (liquidaciones) y el delta muestra absorción: entrada en contra del barrido. |
| **Absorción** | Delta fuerte en una dirección sin que el precio se mueva: alguien absorbe el flujo agresivo. Se confirma con divergencia de CVD. |
| **Funding reversion** | Funding extremo (z-score) indica mercado apalancado hacia un lado: entrada contraria. |
| **OI divergence** | El OI sube mientras el precio cae: cortos abiertos como cobertura mientras se vende spot. |
| **OI drop** | Caída brusca de OI con movimiento fuerte: la cascada de liquidaciones ya ha pasado y el precio tiende a revertir. |
| **Vol compression** | Bollinger dentro de Keltner y ATR comprimido con OI acumulándose: entrada en la ruptura con confirmación de delta. |

El **motor de confluencia** suma las señales con sus pesos y solo opera si la puntuación supera
un mínimo. Por encima de un segundo umbral, la operación se considera reforzada y puede llevar
más tamaño.

## Motor de backtest

**Híbrido vectorizado + vela a vela.** Features y señales se calculan vectorizadas con
pandas/NumPy sobre toda la serie. La gestión de posiciones va vela a vela porque necesita estado:
salidas parciales, trailing stop y stop por tiempo.

**Reglas anti-lookahead**, aplicadas en el código y comprobadas después:
1. Todas las features derivadas se desplazan una vela (`shift(1)`) en `builder.py`: la feature en T solo usa datos hasta T−1.
2. Señal en la vela T → la confirmación se busca desde T+1.
3. Entrada al **open** de la vela de entrada.
4. Stop y TP se comprueban contra el máximo y mínimo de la vela, y **el stop antes que el TP**: si los dos caben en la misma vela, se asume el peor caso.

`LookAheadBiasChecker` revisa cada operación al terminar (momento de entrada, precio de entrada
dentro del rango de la vela, precio de salida coherente) y lista las violaciones.

**Cuatro versiones de gestión** sobre las mismas señales: V1 cierra todo en TP1; V2 cierra parcial
en TP1 y deja correr el resto; V3 añade trailing stop por ATR; V4 añade tamaño reforzado por
confluencia.

**Riesgo y ejecución:** riesgo fijo por operación (0,5 % del equity), stop a 1,5 × ATR, TP1 a
1,5 R y TP2 a 3 R, comisión taker del 0,04 % y slippage del 0,01 %, máximo de 3 operaciones
diarias y parada tras 2 pérdidas seguidas.

## Validación

- **Walk-forward:** entrena en un bloque de 2 meses, valida en el mes siguiente y avanza. Todas las métricas que cuentan salen de periodos que el sistema no ha visto al ajustarse.
- **Monte Carlo:** 10.000 remuestreos con reemplazo de los resultados de las operaciones (semilla fija, reproducible) para estimar la **probabilidad de pasar el challenge** con un 6 % y un 10 % de drawdown máximo.
- **Métricas:** Sharpe, Sortino, Calmar, drawdown máximo y su duración, win rate, profit factor, expectativa y R medio por operación, motivos de salida y si se habría pasado el challenge.

## Ingeniería

- **Descarga concurrente:** los aggTrades se descargan y agregan día a día con un `ThreadPoolExecutor` (4 hilos por defecto). Es trabajo limitado por E/S de red, así que los hilos funcionan bien en Python pese al GIL.
- **Procesado por lotes:** se trabaja en lotes de 20 días y cada lote se agrega a 5 min antes de juntarse, de modo que nunca hay en memoria más de 20 días de ticks en bruto.
- **Parquet con pyarrow** como formato de almacenamiento, con filtrado por fechas al cargar.
- **Interfaz común** (`BaseSignal`) para todas las señales: añadir una señal es una clase nueva y un peso en el YAML.

## Limitaciones y siguientes pasos

- **Escritura incremental de Parquet.** `append_parquet` relee y reescribe el fichero entero en cada lote, así que el coste crece con cada día descargado. Además, los lotes se siguen acumulando en memoria para devolverlos al final. Lo correcto es escribir cada lote como un fichero aparte de un dataset Parquet particionado por fecha y no retener los lotes.
- **Monte Carlo en bucle de Python.** Las 10.000 simulaciones recorren las operaciones una a una. Vectorizándolo con NumPy (matriz de simulaciones × operaciones con `cumsum` y máscaras de drawdown) o paralelizando con `multiprocessing` bajaría el tiempo en órdenes de magnitud.
- **Sin tests unitarios.** `pytest` está en las dependencias pero no hay tests. El simulador de salidas (stop y TP en la misma vela, trailing, parciales) y el detector de lookahead son los primeros candidatos.
- **Un solo activo.** El motor está pensado para un símbolo. La extensión natural es un universo de perpetuos con una cartera compartida.

## Uso

```bash
pip install -r requirements.txt

python -m scripts.download_historical                   # descarga todo a data/raw/
python -m scripts.download_historical --skip-aggtrades  # sin los ticks (mucho más rápido)

python -m scripts.run_backtest                  # V1
python -m scripts.run_backtest --version v3     # V3 con trailing
python -m scripts.run_backtest --walk-forward   # validación walk-forward
python -m scripts.run_backtest --monte-carlo    # probabilidad de pasar el challenge
python -m scripts.run_backtest --all            # todo
```

## Stack

Python · pandas · NumPy · pyarrow (Parquet) · requests · concurrent.futures · PyYAML ·
matplotlib / Plotly
