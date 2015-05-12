import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#input: Pfad zu einer Matrix mit Delta-Werten, als .csv-Datei
#anonym: Autor-Ersatz bei unbekanntem Text, zb. "unbekannt", "anonym", ...
def ingroup_boxplot(data_path, anonym="unbekannt"):
    data = pd.DataFrame().from_csv(data_path, sep=",", encoding="cp1252") #encoding="utf-8"
    #Autorname ist Teil des Dateinamens vor dem Unterstrich
    authors = list(set([a.split("_")[0] for a in data.columns if not a.split("_")[0] == anonym]))
    ingroup_deltas = []
    for i in range(0, len(authors)):
        #alle Werke eines Autors ermitteln
        works = [c for c in data.columns if authors[i] in c]
        #alle ingroup-deltas ermitteln und Nullen entfernen
        values = list(set(data.loc[works, works].values.flatten()))
        values = [val for val in values if val != 0.0]
        ingroup_deltas.append(values)

    #unbekannten Text zu Ingroup-Werten dazunehmen
    ingroup_deltas_anonym = []
    for i in range(0, len(authors)):
        works = [c for c in data.columns if authors[i] in c or anonym in c]
        values = list(set(data.loc[works, works].values.flatten()))
        values = [val for val in values if val != 0.0]
        ingroup_deltas_anonym.append(values)

    #Positionen für die Boxen
    pos = list(range(1, len(authors)+1))
    params = ["boxes", "medians", "whiskers", "caps", "fliers", "means"]
    box1 = plt.boxplot(ingroup_deltas, widths=0.3, positions=[p-0.2 for p in pos])
    box2 = plt.boxplot(ingroup_deltas_anonym, widths=0.3, positions=[p+0.2 for p in pos])
    #Farben für die Boxen
    for p in params:
        plt.setp(box1[p], color="blue")
        plt.setp(box2[p], color="red")
    #Achseneigenschaften
    plt.xticks(np.arange(1, len(authors)+1), authors)
    plt.xlim(xmin=0.5)
    plt.show()
    #plt.savefig("boxplot.png")


ingroup_boxplot("C:/Users/Isabella Reger/Documents/pydelta-master/corpus_deltas/Classic_Delta.2000.case_insensitive.csv")
