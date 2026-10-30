import pandas as pd

LEARN_END = "2024-10-25 23:00"
TEST_END = "2025-10-25 23:00"

def process_database():
    #lecture initiale de la database
    df = pd.read_csv(
        "../../data/data_tarik.csv", sep=";", encoding="utf-8-sig",
        usecols=["Date de début", "Polluant", "type de valeur", "valeur", "validité"],
        parse_dates=["Date de début"],
    )
    df = df[(df["Polluant"] == "NO2") & (df["type de valeur"] == "moyenne horaire validée")]
    return (
        df[["Date de début", "valeur", "validité"]]
        .rename(columns={"Date de début": "date", "valeur": "value", "validité": "validity"})
        .set_index("date")
        .sort_index()
    )

def interpolate(s , max_gap=24):
    #Interpole linéairement les trous de max_gap heures au plus
    missing = s.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    interp = s.interpolate(method="time", limit_area="inside")
    return s.where(~(missing & (run_len <= max_gap)), interp)

def get_data():
    df = process_database()

    s = df["value"].asfreq("h")
    s = interpolate(s)

    train = s[s.index <= LEARN_END]
    val = s[(s.index > LEARN_END) & (s.index <= TEST_END)]
    test = s[s.index > TEST_END]

    return s, train, val, test