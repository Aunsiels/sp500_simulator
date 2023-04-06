import datetime
import json
from calendar import monthrange

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import cpi


RECESSIONS = [(pd.to_datetime("2020-02-01", utc=True), pd.to_datetime("2020-04-01", utc=True)),  # Covid
              (pd.to_datetime("2007-12-01", utc=True), pd.to_datetime("2009-06-01", utc=True)),  # Great Recession
              (pd.to_datetime("2001-03-01", utc=True), pd.to_datetime("2001-11-01", utc=True)),  # Dot Bomb
              (pd.to_datetime("1990-07-01", utc=True), pd.to_datetime("1991-03-01", utc=True)),  # Gulf War
              (pd.to_datetime("1981-07-01", utc=True), pd.to_datetime("1982-11-01", utc=True)),  # Double-Dip Recessin
              (pd.to_datetime("1980-01-01", utc=True), pd.to_datetime("1980-07-01", utc=True)),  # Iran and Volcker
              (pd.to_datetime("1973-11-01", utc=True), pd.to_datetime("1975-03-01", utc=True)),  # Oil Embargo
              ]


def get_livret_A():
    df = pd.read_csv("data/livretA.csv")
    df = df.append({"date": datetime.datetime.now().strftime("%d/%m/%Y"), "interest_rate": np.NAN}, ignore_index=True)
    df["date"] = pd.to_datetime(df.date, dayfirst=True)
    res = dict()
    for _, row in df.resample("1m", on="date").max().fillna(method='ffill').reset_index().iterrows():
        res[row["date"].strftime("%m/%Y")] = row["interest_rate"]
    return res


LIVRET_A = get_livret_A()


def is_recession(date, delta=pd.Timedelta(days=30*0)):
    for begin, end in RECESSIONS:
        if begin + delta < date <= end + delta:
            return True
    return False


AVERAGE_NUMBER_OF_CIGARETTES = 13 / 20  # Proportion size


def compute_values(init_value, monthly_investment_end, variations, number_months,
                   adjust_inflation=False, buy_one_action=False, invest_every=1, cigarette=True):
    if adjust_inflation:
        init_value = cpi.inflate(init_value, 2022, to=variations.iloc[-number_months - 1]["Date"].year)
    res = [(variations.iloc[-number_months - 1]["Date"], init_value, init_value,
            init_value, init_value, init_value, init_value)]
    cigarettes = json.load(open("data/cigarettes.json"))
    prev_high = -1
    for idx, row in variations.iloc[-number_months:].iterrows():
        date = row["Date"]
        year = str(date.year)
        month_year = date.strftime("%m/%Y")
        n_days = monthrange(date.year, date.month)[1]
        if cigarette:
            monthly_investment_end = cigarettes[year] * n_days * AVERAGE_NUMBER_OF_CIGARETTES
        if adjust_inflation and date.year < 2023:
            monthly_investment = cpi.inflate(monthly_investment_end, 2022, to=date.year)
        else:
            monthly_investment = monthly_investment_end
        if buy_one_action:
            monthly_investment = row["close"]
        monthly_investment *= invest_every
        x = row["var"]
        _, with_sp500_prev, without_sp500_prev, with_recession_prev, with_fear, with_fear_minor, livret_a = res[-1]
        if idx % invest_every == 0:
            with_sp = with_sp500_prev * x + monthly_investment
            without_sp = without_sp500_prev + monthly_investment
            if is_recession(date):
                with_recession_prev = with_recession_prev * x  # + monthly_investment
            else:
                with_recession_prev = with_recession_prev * x + monthly_investment
            if row["close"] > prev_high:
                with_fear = with_fear * x + monthly_investment
            else:
                with_fear = with_fear * x
            if x >= 1:
                with_fear_minor = with_fear_minor * x + monthly_investment
            else:
                with_fear_minor = with_fear_minor * x
            if month_year in LIVRET_A:
                livret_a = livret_a * (1 + LIVRET_A[month_year] / 12 / 100) + monthly_investment
        else:
            with_sp = with_sp500_prev * x
            without_sp = without_sp500_prev
            with_fear = with_fear * x
            with_fear_minor = with_fear_minor * x
            if is_recession(date):
                with_recession_prev = with_recession_prev * x
            else:
                with_recession_prev = with_recession_prev * x
            if month_year in LIVRET_A:
                livret_a = livret_a * (1 + LIVRET_A[month_year] / 12 / 100)
        res.append((date, with_sp, without_sp, with_recession_prev, with_fear, with_fear_minor, livret_a))
        prev_high = max(prev_high, row["close"])
    return res


if __name__ == '__main__':
    df = pd.read_csv("data/sp500.csv")
    df.Date = pd.to_datetime(df.Date, utc=True)
    df = df.resample('M', on='Date').agg(low=("Low", lambda x: x.min()),
                                         high=("High", lambda x: x.max()),
                                         open=("Open", lambda x: x.iloc[0]),
                                         close=("Close", lambda x: x.iloc[-1]),
                                         volume=("Volume", lambda x: sum(x))).reset_index()
    open_val = df["close"]
    open_shift = open_val.shift(1)
    var = open_val / open_shift
    df["var"] = var
    normalize = False
    v = compute_values(0, 1, df, 12 * 30, cigarette=True)
    new_df = pd.DataFrame(v, columns=["Date", "Cigarette money invested on S&P500", "Cigarettes cost",
                                      "No investment during recession",
                                      "No investment below previous high", "No investment when going down",
                                      "Livret A"])

    fig = px.line(new_df, x="Date", y=["Cigarette money invested on S&P500", "Cigarettes cost"],
                  markers=True, title="The Real Cost of Cigarettes")

    fig.update_layout(
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(
                          title='Date',
                          autorange=True,
                          showgrid=False,
                          zeroline=True,
                          showline=True,
                          ticks='',
                          linecolor='black',
                          showticklabels=True,
                          rangemode="tozero"
                      ),
                      yaxis=dict(
                          title='Money in euros',
                          autorange=True,
                          showgrid=False,
                          zeroline=True,
                          showline=True,
                          ticks='',
                          linecolor='black',
                          showticklabels=True,
                          rangemode="tozero"
                      ),
                      legend=dict(x=0, y=1),
                      )
    pio.write_html(fig, file="results/cigarettes.html", auto_open=True)
    fig.show()

    exit(0)

    v = compute_values(0, 1, df, 12 * 20, cigarette=False)
    new_df = pd.DataFrame(v, columns=["Date", "Continuous investment on S&P500", "Investment without interests",
                                      "No investment during recession",
                                      "No investment below previous high", "No investment when going down",
                                      "Livret A"])
    last_value = new_df["Investment without interests"].iloc[-1]
    for x in new_df.columns:
        if x != "Date" and normalize:
            new_df[x] = new_df[x] / last_value
    fig = px.line(new_df, x="Date", y=new_df.columns, markers=True)

    fig.show()
