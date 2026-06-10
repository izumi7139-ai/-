# ============================================
# MRGE GitHub Actions版
# 米国株・期待値最大化分析ツール
#
# 自動実行内容：
# ・米国株を分析
# ・Acceleration / Transition / Persistence TOP10を作成
# ・監視ポートフォリオ候補を作成
# ・dataフォルダにCSV保存
# ・LINEへ通知
# ============================================

import os
import io
import time
import requests
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =====================
# 設定
# =====================

TEST_MODE = False
TEST_LIMIT = 500

MIN_MARKET_CAP = 1_000_000_000
MIN_DOLLAR_VOLUME = 1_000_000
SLEEP_SEC = 0.12

TOP_ENGINE = 10

ACC_PORTFOLIO_N = 4
TRANS_PORTFOLIO_N = 3
PERS_PORTFOLIO_N = 3

DATA_DIR = "data"

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")


# =====================
# 基本関数
# =====================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def safe_get(df, names, col):
    if df is None or df.empty:
        return np.nan

    for n in names:
        if n in df.index:
            try:
                row = df.loc[n]
                if len(row) > col:
                    return row.iloc[col]
            except Exception:
                return np.nan

    return np.nan


def pct_change_safe(new, old):
    if pd.isna(new) or pd.isna(old) or old == 0:
        return np.nan
    return (new - old) / abs(old)


def clamp(x, low=0, high=100):
    if pd.isna(x):
        return 0
    return max(low, min(high, x))


def cap_growth(x):
    if pd.isna(x):
        return np.nan
    return min(x, 3.0)


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.iloc[-1]


def score_band(x, bands):
    if pd.isna(x):
        return 0

    for threshold, score in bands:
        if x >= threshold:
            return score

    return 0


# =====================
# 米国株ユニバース
# =====================

def get_us_universe():
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    ]

    dfs = []

    nasdaq_text = requests.get(urls[0], timeout=30).text
    nasdaq = pd.read_csv(io.StringIO(nasdaq_text), sep="|")
    nasdaq = nasdaq[nasdaq["Symbol"].notna()]
    nasdaq = nasdaq[nasdaq["Symbol"] != "File Creation Time"]
    nasdaq["ticker"] = nasdaq["Symbol"]
    nasdaq["name"] = nasdaq["Security Name"]
    nasdaq["is_etf"] = nasdaq["ETF"]
    dfs.append(nasdaq[["ticker", "name", "is_etf"]])

    other_text = requests.get(urls[1], timeout=30).text
    other = pd.read_csv(io.StringIO(other_text), sep="|")
    other = other[other["ACT Symbol"].notna()]
    other = other[other["ACT Symbol"] != "File Creation Time"]
    other["ticker"] = other["ACT Symbol"]
    other["name"] = other["Security Name"]
    other["is_etf"] = other["ETF"]
    dfs.append(other[["ticker", "name", "is_etf"]])

    df = pd.concat(dfs, ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    df["name"] = df["name"].astype(str)

    df = df[df["is_etf"] == "N"]

    exclude_words = [
        "ETF", "ETN", "Fund", "Closed End", "Income Trust",
        "REIT", "Real Estate Investment Trust",
        "ADR", "ADS", "American Depositary", "Depositary",
        "Acquisition", "SPAC", "Warrant", "Right", "Unit",
        "Gold", "Silver", "Mining", "Mineral",
        "Biotechnology", "BioPharma", "Pharmaceutical",
        "Therapeutics", "Therapy", "Clinical", "Oncology"
    ]

    pattern = "|".join(exclude_words)
    df = df[~df["name"].str.contains(pattern, case=False, na=False)]

    return df.drop_duplicates("ticker").reset_index(drop=True)


def is_bad_name(name):
    name_lower = str(name).lower()

    bad_words = [
        "american depositary",
        "depositary",
        "adr",
        "ads",
        "reit",
        "real estate investment trust",
        "acquisition",
        "spac",
        "warrant",
        "right",
        "unit",
        "gold",
        "silver",
        "mining",
        "mineral",
        "biotechnology",
        "biopharma",
        "pharmaceutical",
        "therapeutics",
        "oncology",
        "clinical"
    ]

    return any(w in name_lower for w in bad_words)


def is_excluded_sector(sector, industry):
    text = f"{sector} {industry}".lower()

    excluded_words = [
        "biotechnology",
        "pharmaceutical",
        "drug",
        "therapeutic",
        "healthcare",
        "medical care",
        "diagnostics",
        "clinical",
        "reit",
        "real estate investment trust",
        "gold",
        "silver",
        "mining",
        "metals"
    ]

    return any(w in text for w in excluded_words)


# =====================
# テーマ分類
# =====================

def classify_theme(name, sector, industry):
    text = f"{name} {sector} {industry}".lower()

    if any(w in text for w in [
        "semiconductor", "chip", "electronic components",
        "integrated circuits", "silicon", "gpu"
    ]):
        return "半導体"

    if any(w in text for w in [
        "software", "cloud", "saas", "application software",
        "infrastructure software"
    ]):
        return "ソフトウェア"

    if any(w in text for w in [
        "cybersecurity", "security software", "network security"
    ]):
        return "サイバー"

    if any(w in text for w in [
        "aerospace", "defense", "space"
    ]):
        return "防衛/宇宙"

    if any(w in text for w in [
        "energy", "power", "electrical", "electric", "nuclear"
    ]):
        return "電力/エネルギー"

    if any(w in text for w in [
        "automation", "robot", "robotics"
    ]):
        return "ロボティクス"

    if any(w in text for w in [
        "communication equipment", "optical", "networking"
    ]):
        return "通信/光通信"

    return "その他"


# =====================
# 各スコア
# =====================

def acceleration_score(g1, g2, g3, gross_margin, fcf_growth, op_margin_now, op_margin_prev):
    score = 0

    g1 = cap_growth(g1)
    g2 = cap_growth(g2)
    g3 = cap_growth(g3)
    fcf_growth = cap_growth(fcf_growth)

    growths = [g3, g2, g1]

    if not any(pd.isna(x) for x in growths):
        if growths[2] > growths[1] > growths[0]:
            score += 30
        elif growths[2] > growths[1]:
            score += 18

    score += score_band(g1, [
        (1.00, 25),
        (0.70, 22),
        (0.50, 18),
        (0.30, 14),
        (0.20, 9),
        (0.10, 5)
    ])

    score += score_band(gross_margin, [
        (0.75, 15),
        (0.60, 12),
        (0.45, 8),
        (0.30, 4)
    ])

    score += score_band(fcf_growth, [
        (1.00, 12),
        (0.50, 9),
        (0.20, 5),
        (0.00, 2)
    ])

    if not pd.isna(op_margin_now) and not pd.isna(op_margin_prev):
        improvement = op_margin_now - op_margin_prev
        if improvement >= 0.10:
            score += 10
        elif improvement >= 0.05:
            score += 6
        elif improvement >= 0.02:
            score += 3

    return clamp(score)


def transition_score(net_now, net_prev, op_margin_now, op_margin_prev, fcf_now, fcf_prev, revenue_growth):
    score = 0

    revenue_growth = cap_growth(revenue_growth)

    if not pd.isna(net_now) and not pd.isna(net_prev):
        if net_prev < 0 and net_now > 0:
            score += 35
        elif net_now > net_prev and net_now < 0:
            score += 18
        elif net_now > net_prev:
            score += 12

    if not pd.isna(op_margin_now) and not pd.isna(op_margin_prev):
        improvement = op_margin_now - op_margin_prev
        if improvement >= 0.20:
            score += 30
        elif improvement >= 0.12:
            score += 22
        elif improvement >= 0.06:
            score += 14
        elif improvement >= 0.03:
            score += 7

    if not pd.isna(fcf_now) and not pd.isna(fcf_prev):
        if fcf_prev < 0 and fcf_now > 0:
            score += 25
        elif fcf_now > fcf_prev:
            score += 12

    score += score_band(revenue_growth, [
        (0.50, 10),
        (0.30, 8),
        (0.20, 5),
        (0.10, 2)
    ])

    return clamp(score)


def persistence_score(g1, g2, g3, gross_margin, fcf_now, fcf_prev, debt_ratio):
    score = 0

    g1 = cap_growth(g1)
    g2 = cap_growth(g2)
    g3 = cap_growth(g3)

    growths = [g1, g2, g3]
    valid_growths = [x for x in growths if not pd.isna(x)]

    if len(valid_growths) == 3:
        if all(x >= 0.15 for x in valid_growths):
            score += 30
        elif all(x >= 0.10 for x in valid_growths):
            score += 22
        elif all(x > 0 for x in valid_growths):
            score += 12

        avg_growth = np.mean(valid_growths)
        std_growth = np.std(valid_growths)

        if avg_growth >= 0.20 and std_growth <= 0.15:
            score += 25
        elif avg_growth >= 0.15 and std_growth <= 0.25:
            score += 15
        elif avg_growth >= 0.10:
            score += 8

    score += score_band(gross_margin, [
        (0.75, 15),
        (0.60, 12),
        (0.45, 8),
        (0.30, 4)
    ])

    if not pd.isna(fcf_now) and fcf_now > 0:
        score += 15
        if not pd.isna(fcf_prev) and fcf_now > fcf_prev:
            score += 5

    if not pd.isna(debt_ratio):
        if debt_ratio < 0.25:
            score += 10
        elif debt_ratio < 0.40:
            score += 6
        elif debt_ratio < 0.60:
            score += 2

    return clamp(score)


def upside_score(market_cap):
    if pd.isna(market_cap):
        return 0

    if market_cap < 3_000_000_000:
        return 100
    if market_cap < 10_000_000_000:
        return 85
    if market_cap < 30_000_000_000:
        return 65
    if market_cap < 100_000_000_000:
        return 40
    if market_cap < 300_000_000_000:
        return 25
    return 10


def entry_score(rsi, gap50, dist_high, above200):
    score = 0

    if not pd.isna(rsi):
        if 40 <= rsi <= 60:
            score += 35
        elif 30 <= rsi < 40:
            score += 25
        elif 60 < rsi <= 70:
            score += 15
        elif rsi < 30:
            score += 8

    if not pd.isna(gap50):
        if -0.05 <= gap50 <= 0.05:
            score += 30
        elif -0.12 <= gap50 < -0.05:
            score += 20
        elif 0.05 < gap50 <= 0.10:
            score += 15
        elif -0.25 <= gap50 < -0.12:
            score += 8

    if not pd.isna(dist_high):
        if 0.08 <= dist_high <= 0.25:
            score += 25
        elif 0.03 <= dist_high < 0.08:
            score += 12
        elif 0.25 < dist_high <= 0.40:
            score += 8

    if above200:
        score += 10

    score = clamp(score)

    if score >= 80:
        grade = "S"
    elif score >= 65:
        grade = "A"
    elif score >= 50:
        grade = "B"
    elif score >= 35:
        grade = "C"
    else:
        grade = "D"

    return score, grade


def risk_penalty(name, sector, industry, debt_ratio, op_cf_now, market_cap):
    text = f"{name} {sector} {industry}".lower()

    hard_exclude_words = [
        "american depositary",
        "depositary",
        "adr",
        "ads",
        "biotechnology",
        "biopharma",
        "pharmaceutical",
        "therapeutic",
        "drug",
        "oncology",
        "clinical",
        "healthcare",
        "medical",
        "diagnostics",
        "reit",
        "real estate investment trust",
        "gold",
        "silver",
        "mining",
        "metals"
    ]

    if any(w in text for w in hard_exclude_words):
        return -999

    penalty = 0

    soft_penalty_words = [
        "food",
        "egg",
        "farm",
        "agriculture",
        "beverage",
        "restaurant",
        "grocery",
        "commodity"
    ]

    if any(w in text for w in soft_penalty_words):
        penalty -= 20

    financial_words = [
        "bank",
        "financial",
        "insurance",
        "mortgage",
        "credit services",
        "asset management"
    ]

    if any(w in text for w in financial_words):
        penalty -= 20

    if not pd.isna(debt_ratio):
        if debt_ratio > 0.80:
            penalty -= 20
        elif debt_ratio > 0.60:
            penalty -= 10

    if not pd.isna(op_cf_now) and op_cf_now < 0:
        penalty -= 10

    if not pd.isna(market_cap) and market_cap < 2_000_000_000:
        penalty -= 5

    return penalty


# =====================
# 1銘柄分析
# =====================

def analyze_ticker(ticker, name):
    try:
        if is_bad_name(name):
            return None

        stock = yf.Ticker(ticker)
        info = stock.info

        market_cap = info.get("marketCap", np.nan)
        sector = info.get("sector", "")
        industry = info.get("industry", "")

        country = info.get("country", "")
        if country and country != "United States":
            return None

        if is_excluded_sector(sector, industry):
            return None

        if pd.isna(market_cap) or market_cap < MIN_MARKET_CAP:
            return None

        hist = stock.history(period="1y", auto_adjust=True)

        if hist.empty or len(hist) < 220:
            return None

        close = hist["Close"]
        volume = hist["Volume"]

        price = close.iloc[-1]
        avg_vol20 = volume.tail(20).mean()
        dollar_volume = price * avg_vol20

        if dollar_volume < MIN_DOLLAR_VOLUME:
            return None

        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        high_52w = close.max()

        rsi = calc_rsi(close)
        gap50 = (price - ma50) / ma50 if ma50 > 0 else np.nan
        dist_high = (high_52w - price) / high_52w if high_52w > 0 else np.nan
        above200 = price > ma200 if not pd.isna(ma200) else False

        fin = stock.financials
        bal = stock.balance_sheet
        cf = stock.cashflow

        rev = [safe_get(fin, ["Total Revenue"], i) for i in range(4)]
        gross_profit = safe_get(fin, ["Gross Profit"], 0)

        net_income = [safe_get(fin, ["Net Income"], i) for i in range(2)]
        operating_income = [safe_get(fin, ["Operating Income"], i) for i in range(2)]

        operating_cf = [
            safe_get(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"], i)
            for i in range(2)
        ]

        capex = [
            safe_get(cf, ["Capital Expenditure", "Capital Expenditures"], i)
            for i in range(2)
        ]

        fcf_now = (
            operating_cf[0] + capex[0]
            if not pd.isna(operating_cf[0]) and not pd.isna(capex[0])
            else np.nan
        )

        fcf_prev = (
            operating_cf[1] + capex[1]
            if not pd.isna(operating_cf[1]) and not pd.isna(capex[1])
            else np.nan
        )

        total_debt = safe_get(bal, ["Total Debt"], 0)
        total_assets = safe_get(bal, ["Total Assets"], 0)

        debt_ratio = (
            total_debt / total_assets
            if not pd.isna(total_debt) and not pd.isna(total_assets) and total_assets > 0
            else np.nan
        )

        g1 = pct_change_safe(rev[0], rev[1])
        g2 = pct_change_safe(rev[1], rev[2])
        g3 = pct_change_safe(rev[2], rev[3])

        gross_margin = (
            gross_profit / rev[0]
            if not pd.isna(gross_profit) and not pd.isna(rev[0]) and rev[0] > 0
            else np.nan
        )

        fcf_growth = pct_change_safe(fcf_now, fcf_prev)

        op_margin_now = (
            operating_income[0] / rev[0]
            if not pd.isna(operating_income[0]) and not pd.isna(rev[0]) and rev[0] > 0
            else np.nan
        )

        op_margin_prev = (
            operating_income[1] / rev[1]
            if not pd.isna(operating_income[1]) and not pd.isna(rev[1]) and rev[1] > 0
            else np.nan
        )

        risk = risk_penalty(name, sector, industry, debt_ratio, operating_cf[0], market_cap)

        if risk <= -999:
            return None

        acc = acceleration_score(
            g1, g2, g3, gross_margin, fcf_growth, op_margin_now, op_margin_prev
        )

        trans = transition_score(
            net_income[0], net_income[1],
            op_margin_now, op_margin_prev,
            fcf_now, fcf_prev,
            g1
        )

        pers = persistence_score(
            g1, g2, g3, gross_margin, fcf_now, fcf_prev, debt_ratio
        )

        up = upside_score(market_cap)
        ent, ent_grade = entry_score(rsi, gap50, dist_high, above200)

        acceleration_final = clamp(acc * 0.75 + up * 0.15 + ent * 0.10 + risk)
        transition_final = clamp(trans * 0.75 + up * 0.15 + ent * 0.10 + risk)
        persistence_final = clamp(pers * 0.55 + up * 0.25 + ent * 0.20 + risk)

        total_score = max(acceleration_final, transition_final, persistence_final)

        if total_score == acceleration_final:
            best_type = "Acceleration"
        elif total_score == transition_final:
            best_type = "Transition"
        else:
            best_type = "Persistence"

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "country": country,
            "theme": classify_theme(name, sector, industry),
            "price": price,
            "market_cap_bil": market_cap / 1_000_000_000,
            "dollar_volume_mil": dollar_volume / 1_000_000,
            "revenue_growth_1y": cap_growth(g1),
            "revenue_growth_2y": cap_growth(g2),
            "revenue_growth_3y": cap_growth(g3),
            "gross_margin": gross_margin,
            "fcf_growth": cap_growth(fcf_growth),
            "debt_ratio": debt_ratio,
            "rsi": rsi,
            "gap50": gap50,
            "dist_high": dist_high,
            "above200": above200,
            "acceleration_score": acc,
            "transition_score": trans,
            "persistence_score": pers,
            "upside_score": up,
            "entry_score": ent,
            "entry_grade": ent_grade,
            "risk_penalty": risk,
            "acceleration_final": acceleration_final,
            "transition_final": transition_final,
            "persistence_final": persistence_final,
            "total_score": total_score,
            "best_type": best_type
        }

    except Exception as e:
        return None


# =====================
# LINE送信
# =====================

def send_line_message(message):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("LINE secrets are not set. Skip LINE notification.")
        return

    url = "https://api.line.me/v2/bot/message/push"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }

    data = {
        "to": LINE_USER_ID,
        "messages": [
            {
                "type": "text",
                "text": message[:4900]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)

    print(f"LINE送信ステータス: {response.status_code}")
    if response.status_code >= 300:
        print(response.text)


# =====================
# メイン
# =====================

def main():
    ensure_data_dir()

    print("米国株ユニバース取得中...")
    universe = get_us_universe()
    print(f"取得銘柄数: {len(universe)}")

    if TEST_MODE:
        universe = universe.head(TEST_LIMIT)
        print(f"TEST_MODE=True: {TEST_LIMIT}銘柄のみ分析")
    else:
        print("全銘柄分析モード")

    results = []

    for _, row in tqdm(universe.iterrows(), total=len(universe)):
        result = analyze_ticker(row["ticker"], row["name"])
        if result is not None:
            results.append(result)
        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(results)

    if df.empty:
        message = "【MRGE】条件に合う銘柄がありませんでした。"
        print(message)
        send_line_message(message)
        return

    today = datetime.now().strftime("%Y%m%d")

    all_path = os.path.join(DATA_DIR, f"mrge_v60_all_{today}.csv")
    df.to_csv(all_path, index=False)

    acc_top = df.sort_values("acceleration_final", ascending=False).head(TOP_ENGINE).copy()
    trans_top = df.sort_values("transition_final", ascending=False).head(TOP_ENGINE).copy()
    pers_top = df.sort_values("persistence_final", ascending=False).head(TOP_ENGINE).copy()

    acc_top.insert(0, "rank", range(1, len(acc_top) + 1))
    trans_top.insert(0, "rank", range(1, len(trans_top) + 1))
    pers_top.insert(0, "rank", range(1, len(pers_top) + 1))

    acc_top.to_csv(os.path.join(DATA_DIR, f"mrge_v60_acceleration_{today}.csv"), index=False)
    trans_top.to_csv(os.path.join(DATA_DIR, f"mrge_v60_transition_{today}.csv"), index=False)
    pers_top.to_csv(os.path.join(DATA_DIR, f"mrge_v60_persistence_{today}.csv"), index=False)

    portfolio = pd.concat([
        acc_top.head(ACC_PORTFOLIO_N),
        trans_top.head(TRANS_PORTFOLIO_N),
        pers_top.head(PERS_PORTFOLIO_N)
    ]).drop_duplicates("ticker")

    portfolio = portfolio.sort_values("total_score", ascending=False).reset_index(drop=True)
    portfolio.insert(0, "portfolio_rank", range(1, len(portfolio) + 1))

    portfolio.to_csv(os.path.join(DATA_DIR, f"mrge_v60_portfolio_{today}.csv"), index=False)

    history_path = os.path.join(DATA_DIR, "mrge_history.csv")

    history_add = portfolio.copy()
    history_add["snapshot_date"] = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(history_path):
        old_history = pd.read_csv(history_path)
        history = pd.concat([old_history, history_add], ignore_index=True)
        history = history.drop_duplicates(subset=["snapshot_date", "ticker"], keep="last")
    else:
        history = history_add

    history.to_csv(history_path, index=False)

    message = "【MRGE 米国株期待値最大化ランキング】\n\n"

    message += "■ Acceleration TOP10\n"
    for _, r in acc_top.iterrows():
        message += (
            f"{int(r['rank'])}位 {r['ticker']}｜"
            f"{r['acceleration_final']:.0f}点｜"
            f"Entry {r['entry_grade']}｜"
            f"{r['theme']}\n"
        )

    message += "\n■ Transition TOP10\n"
    for _, r in trans_top.iterrows():
        message += (
            f"{int(r['rank'])}位 {r['ticker']}｜"
            f"{r['transition_final']:.0f}点｜"
            f"Entry {r['entry_grade']}｜"
            f"{r['theme']}\n"
        )

    message += "\n■ Persistence TOP10\n"
    for _, r in pers_top.iterrows():
        message += (
            f"{int(r['rank'])}位 {r['ticker']}｜"
            f"{r['persistence_final']:.0f}点｜"
            f"Entry {r['entry_grade']}｜"
            f"{r['theme']}\n"
        )

    message += "\n■ 監視ポートフォリオ候補\n"
    for _, r in portfolio.iterrows():
        message += (
            f"{int(r['portfolio_rank'])}. {r['ticker']}｜"
            f"{r['best_type']}｜"
            f"総合{r['total_score']:.0f}｜"
            f"Entry {r['entry_grade']}｜"
            f"{r['theme']}\n"
        )

    print(message)
    send_line_message(message)


if __name__ == "__main__":
    main()
