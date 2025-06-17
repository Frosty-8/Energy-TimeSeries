import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

rprint(Panel("[bold blue]Starting Energy Consumption Analysis[/bold blue]", expand=False))

with console.status("[green]Loading data...[/green]", spinner="dots"):
    df = pd.read_csv('dataset/PJME_hourly.csv')
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)
console.log("[green]Data loaded and preprocessed.[/green]")

df.plot(style='.',figsize=(12,6),color=color_pal[1],title='Hourly Energy consumption')
plt.show()
rprint("[bold yellow]Displayed hourly energy consumption plot.[/bold yellow]")

df['PJME_MW'].plot(kind='hist',bins=500)
plt.show()
rprint("[bold yellow]Displayed histogram of PJME_MW.[/bold yellow]")

df.query('PJME_MW < 19000')['PJME_MW'].plot(style='.',figsize=(12,6),color = color_pal[2],title='Outliers')
plt.show()
rprint("[bold yellow]Displayed outliers plot.[/bold yellow]")

df = df.query('PJME_MW > 19000').copy()
console.log("[green]Outliers removed from data.[/green]")

train = df.loc[df.index < '01-01-2014']
test = df.loc[df.index >= '01-01-2014']

fig,ax = plt.subplots(figsize=(12,6))
train.plot(ax=ax,title='Training set')
test.plot(ax=ax,title='Test set')
ax.axvline('01-01-2014',color='black',ls='--')
ax.legend(['Training set','Test set'])
plt.show()
rprint("[bold yellow]Displayed initial train/test split.[/bold yellow]")

tsx = TimeSeriesSplit(n_splits=6,test_size=24*365*1,gap=24*30)
df = df.sort_index()

fig , axs = plt.subplots(6,1,figsize=(12,12),sharex=True)

fold_display = 0
with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task_tsc_plot = progress.add_task("[cyan]Generating TimeSeriesSplit plots...", total=6)
    for train_idx , val_idx in tsx.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]
        train['PJME_MW'].plot(ax=axs[fold_display],label='Training set',title=f'Fold {fold_display}')
        test['PJME_MW'].plot(ax=axs[fold_display],label='Testing set')
        axs[fold_display].axvline(test.index.min(),color='black',ls='--')
        fold_display += 1
        progress.update(task_tsc_plot, advance=1)
plt.tight_layout()
plt.show()
rprint("[bold yellow]Displayed TimeSeriesSplit plots.[/bold yellow]")

# Removed @console.status decorator from here
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Apply console.status when calling the function
with console.status("[green]Creating features...[/green]", spinner="line"):
    df = create_features(df)
console.log("[green]Features created.[/green]")

# Removed @console.status decorator from here
def add_lags(df):
    target_map = df['PJME_MW'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

# Apply console.status when calling the function
with console.status("[green]Adding lag features...[/green]", spinner="line"):
    df = add_lags(df)
console.log("[green]Lag features added.[/green]")

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()
fold = 0
preds = []
scores = []

rprint(Panel("[bold green]Starting Model Training with TimeSeriesSplit[/bold green]", expand=False))

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task_training = progress.add_task("[cyan]Training XGBoost model...", total=tss.n_splits)

    for train_idx, val_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]

        train = create_features(train)
        test = create_features(test)

        FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                    'lag1','lag2','lag3']
        TARGET = 'PJME_MW'
        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                               n_estimators=1000,
                               early_stopping_rounds=50,
                               objective='reg:linear',
                               max_depth=3,
                               learning_rate=0.01)
        progress.console.log(f"  [italic grey50]Fold {fold}: Fitting model...[/italic grey50]")
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False)

        y_pred = reg.predict(X_test)
        preds.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)
        progress.console.log(f"  [green]Fold {fold} finished. RMSE: {score:0.4f}[/green]")
        fold += 1
        progress.update(task_training, advance=1)

table = Table(title="Model Performance Across Folds")
table.add_column("Fold", style="cyan", no_wrap=True)
table.add_column("RMSE", style="magenta")

for i, score in enumerate(scores):
    table.add_row(str(i), f"{score:0.4f}")

console.print(table)
rprint(f'Score across folds [bold blue]{np.mean(scores):0.4f}[/bold blue]')

rprint(Panel("[bold green]Training Final Model and Making Future Predictions[/bold green]", expand=False))

df = create_features(df)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
            'lag1','lag2','lag3']
TARGET = 'PJME_MW'

X_all = df[FEATURES]
y_all = df[TARGET]

with console.status("[green]Training final XGBoost model on all data...[/green]", spinner="bounce"):
    reg = xgb.XGBRegressor(base_score=0.5,
                           booster='gbtree',
                           n_estimators=500,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_all, y_all,
            eval_set=[(X_all, y_all)],
            verbose=False)
console.log("[green]Final model trained.[/green]")

future = pd.date_range('2018-08-03','2019-08-01', freq='1h')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
df_and_future = pd.concat([df, future_df])
df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)

future_w_features = df_and_future.query('isFuture').copy()

with console.status("[green]Generating future predictions...[/green]", spinner="triangle"):
    future_w_features['pred'] = reg.predict(future_w_features[FEATURES])
console.log("[green]Future predictions generated.[/green]")

future_w_features['pred'].plot(figsize=(10, 5),
                               color=color_pal[4],
                               ms=1,
                               lw=1,
                               title='Future Predictions')
plt.show()
rprint("[bold yellow]Displayed future predictions plot.[/bold yellow]")

with console.status("[green]Saving model...[/green]", spinner="dots"):
    reg.save_model('model.json')
console.log("[green]Model saved to model.json.[/green]")

rprint(Panel("[bold blue]Analysis Complete![/bold blue]", expand=False))