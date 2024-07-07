# Experiment 1: Abwechselnd Schwarz Weiß trainieren:

In diesem Experiment initalisieren wir einen Agenten der für Spieler Weiß (Spieler 1) oder Spieler Schwarz (Spieler 2) zieht und dabei trainiert. Der Gegenzug wird hierbei abwechselnd von Random gezogen.

Setup:
- Boardsize = 7*7
- Gegner = Random
- Reward = 
    - 1 bei Gewonnenem Spiel (Wenn er den gewinnenden Zug ausführt, egal ob schwarz oder weiß)
    - 0 Wenn er das Spiel verliert
    - Der Reward wird zusätzlich durch die Anzahl der Zuge beeinflusst (siehe Code Zeile: )

Extra:
  - Reward Verteilung noch anpassen
  - mit Shaping blocking und connecting ist die winrate um 1% niedriger 

Notizen:
wenn agent immer nur ein spieler ist, konvergiert er schneller mit den wenigsten zügen
