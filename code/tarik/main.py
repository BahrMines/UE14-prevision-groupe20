import argparse

from data.data import get_data
from data.export import FIGURES, STAT_NAMES, TABLE, Exports, Plots
from ai.metrics import stats4
from ai.models import difference
from ai.training import compute_metrics, predict_test, select_hyperparameters

SERIES = "NO2 Ajaccio Canetto (Tarik)"
DOMAIN = "Environnement, horaire"


def print_stats(raw: tuple, diff: tuple) -> None:
    print(f"{'':8s}" + "".join(f"{name:>12s}" for name in STAT_NAMES))
    print(f"{'brute':8s}" + "".join(f"{v:12.3f}" for v in raw))
    print(f"{'diff.':8s}" + "".join(f"{v:12.3f}" for v in diff))


def print_results(res: dict, pred: dict) -> None:
    print(f"{'':10s}" + "".join(f"{metric:>10s}" for metric in res))
    for model in pred:
        print(f"{model:10s}" + "".join(f"{res[metric][model]:10.3f}" for metric in res))
    biased = [model for model in pred if abs(res["nmbe"][model]) > 1]
    print(f"|nMBE| > 1 % : {biased}")


def main(show: bool) -> None:
    exports = Exports(SERIES, DOMAIN)
    plots = Plots(show)

    s, train, val, test = get_data()
    x = s.to_numpy()
    i1 = len(train)
    i2 = i1 + len(val)
    for name, part in (("Apprentissage", train), ("Validation", val), ("Test", test)):
        print(f"{name:13s} {part.index[0]} -> {part.index[-1]} ({len(part)} h)")

    print("\n=== 1) description (apprentissage) ===")
    raw, diff = stats4(x[:i1]), stats4(difference(x)[:i1])
    print_stats(raw, diff)
    plots.series(s, i1, i2)

    print("\n=== 2) choix des réglages (validation) ===")
    hp, pers = select_hyperparameters(x, i1, i2)
    plots.validation("AR", hp, pers)
    plots.validation("ELM", hp, pers)

    print("\n=== 3) évaluation (test) ===")
    rows, pred = predict_test(x, i1, i2, hp)
    xobs = x[rows]
    index = s.index[rows]
    print(f"Test : {len(rows)} heures évaluées sur {len(x) - i2} ; moyenne de x = {xobs.mean():.4f}")
    res = compute_metrics(xobs, pred)
    print_results(res, pred)

    exports.save(raw, diff, i1, len(rows), res)
    for name, prefix in (("Persistance", "Pers."), ("AR", "AR"), ("ELM", "ELM")):
        plots.forecast(name, index, xobs, {label: xhat for label, xhat in pred.items() if label.split()[0] == prefix})
    plots.nrmse(res["nrmse"])
    print(f"\nTableau : {TABLE}\nFigures : {FIGURES}")
    plots.display()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prévision à 1 h du NO2")
    parser.add_argument("--show", action="store_true", help="afficher les figures à l'écran")
    main(parser.parse_args().show)