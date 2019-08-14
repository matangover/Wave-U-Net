import music21
import sys

chorale = int(sys.argv[1])
score = music21.corpus.chorales.Iterator(currentNumber=chorale).next()
score.show(app='MuseScore 3')