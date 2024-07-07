# Experiment 1: Abwechselnd Schwarz Weiß trainieren:

In diesem Experiment initalisieren wir einen Agenten der dann abwechselnd für Spieler Weiß (Spieler 1) uns Spieler Schwarz (Spieler 2) zieht und dabei trainiert. Der Gegenzug wird hierbei abwechseln von Random gezogen.

Setup:
- Boardsize = 7*7
- Gegener = Random
- Reward = 
    - 1 bei Gewonnenem Spiel (Wenn er den gewinnenden Zug ausführt, egal ob schwarz oder weiß)
    - 0 Wenn er das Spiel verliert
    - Der Reward wird zusätzlich durch die Anzahl der Zuge beeinflusst (siehe Code Zeile: )



Notizen:
wenn agent immer nur ein spieler ist, konvergiert er schneller mit den wenigsten zügen
bei hotswap gleich abwechselnd, tut er sich schon schwer und wenige züge werden nicht gut gelernt
noch implementieren, wie verschiedene modelle geladen werden können