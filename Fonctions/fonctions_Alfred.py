import numpy as np


def persistance(serie):

    # serie.shift(1) décale les valeurs d’une ligne vers le bas en conservant les dates.
    return serie.shift(1)

def statistiques(serie):
    moyenne = np.mean(serie)
    ecart_type = np.std(serie, ddof=0)

    valeurs_centrees = serie - moyenne

    asymetrie = np.mean(valeurs_centrees ** 3) / ecart_type ** 3
    kurtosis = np.mean(valeurs_centrees ** 4) / ecart_type ** 4

    return {
        "moyenne": moyenne,
        "ecart_type": ecart_type,
        "asymetrie": asymetrie,
        "kurtosis": kurtosis
    }