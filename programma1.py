import sys
import nltk

def parser(frasi):
	#creo due liste per memorizzare i tokens e il PoS tagging
	tokensTot = []
	tokensPoSTot = []
	
	#con un ciclo for scorro tutte le frasi
	for frase in frasi:
		#tokenizzo la frase
		tokens = nltk.word_tokenize(frase)
		tokensPoS = nltk.pos_tag(tokens)
		
		#incremento le due liste
		tokensTot += tokens
		tokensPoSTot += tokensPoS
	
	#mando in output le liste ottenute la lunghezza del corpus
	return tokensTot, len(tokensTot), tokensPoSTot
		
def mediaCaratteri(tokensList, txtLen):
	#creo una variabile per memorizzare il numero di caratteri totali
	caratteriTot = 0
	
	#creo un ciclo che scorra tutti i tokens del corpus
	for token in tokensList:
		#sommo i caratteri ottenuti misurando la lunghezza del token con len(tok)
		caratteriTot += len(token)
	
	#mando in output la media calcolata direttamente nel return
	return caratteriTot / txtLen
	
def analisiClassiFrequenza(tokensList):
	#creo due contatori: 'p' per le porzioni incrementali, e 't' per lo scorrimento dei tokens
	p = 499
	t = 0
	#creo le variabili per memorizzare le 3 classi di frequenza
	V1, V5, V10 = 0, 0, 0
	
	#creo una variabile che verrà aggiornata con il vocabolario di ogni porzione incrementale
	voc = list(set(tokensList[:p]))

	#creo un unico ciclo while che si interrompe nel momento in cui raggiungo l'ultima porzione incrementale
	#il ciclo incrementa sia il contatore 'i' che il contatore 'c'
	while p < 500*(len(tokensList)//500):
		#scorro il vocabolario all'interno di 'voc'
		parolaTipo = voc[t]
		
		#incremento il contatore c, che mi servirà per controllare il passaggio alla porzione incrementale successiva
		t += 1
		
		#controllo la frequenza della parola tipo e assegno la relativa classe di frequenza
		if tokensList[:p].count(parolaTipo) == 1:
			V1 += 1
		elif tokensList[:p].count(parolaTipo) == 5:
			V5 += 1
		elif tokensList[:p].count(parolaTipo) == 10:
			V10 += 1
		
		if t == len(voc):
			#conclusa la porzione incrementale, stampo i valori ottenuti		
			print("| {:>10}{:>10}{:>10}{:>10} |".format(p+1, V1, V5, V10))
		
			#incremento il contatore 'i' per passare alla porzione incrementale successiva
			p += 500
			#resetto il contatore 'c' e tutte le classi di frequenza
			t, V1, V5, V10 = 0, 0, 0, 0
			#ricalcolo il vocabolario della nuova porzione, incrementata di 500 tokens
			voc = list(set(tokensList[:p]))
			
def analisiPoS(PoS, nFrasi):
	#creo 5 diverse variabili, sulla base delle richieste del progetto
	sostantivi = 0
	verbi = 0
	avverbi = 0
	aggettivi = 0
	parole = 0
	
	#avvio un ciclo for che scorra la lista di PoS tagging
	for tag in PoS:
		#controllo che il token non sia una virgola o un punto
		if tag[1] not in [",", "."]:
			#se non lo è incremento il contatore di parole
			parole += 1
			
			#controllo la classe del token per incrementare la variabile della classe corrispondente
			if tag[1] in ["NN", "NNS", "NNP", "NNPS"]:
				sostantivi += 1
			elif tag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
				verbi += 1
			elif tag[1] in ["WRB", "RB", "RBR", "RBS"]:
				avverbi += 1
			elif tag[1] in ["JJ", "JJR", "JJS"]:
				aggettivi += 1
	
	#per la struttura della funzione PoSso calcolare direttamente la densità lessicale
	densLex = (sostantivi + verbi + avverbi + aggettivi) / parole
	
	#mando in output direttamente le medie richieste, insieme alla densità lessicale
	return sostantivi/nFrasi, verbi/nFrasi, densLex
				
def main(joeBiden, donaldTrump):
	#apro e leggo i txt contententi i discorsi di J. Biden e D. Trump
	JBInput = open(joeBiden, mode="r", encoding="utf-8")
	DTInput = open(donaldTrump, mode="r", encoding="utf-8")
	JBString = JBInput.read()
	DTString = DTInput.read()
	
	#divido i testi in frasi
	engPickle = nltk.data.load("tokenizers/punkt/english.pickle")
	frasiJB = engPickle.tokenize(JBString)
	frasiDT = engPickle.tokenize(DTString)
	
	#calcolo il numero di frasi nei discorsi di J. Biden e D. Trump
	nFrasiJB = len(frasiJB)
	nFrasiDT = len(frasiDT)
	
	#tokenizzo il testo e calcolo il numero di token con l'apPoSita funzione, oltre che il PoS tagging
	tokensListJB, JBLen, JBPoS = parser(frasiJB)
	tokensListDT, DTLen, DTPoS = parser(frasiDT)
	
	#stampo il numero di frasi e di tokens per ogni testo
	print("Programma 1 - Confrontate i due testi sulla base delle seguenti informazioni statistiche")
	print("\n1) il numero di frasi e di token\n")
	
	print("-"*79)
	print("| {:^75} |".format("LUNGHEZZA DEI CORPUS"))
	print("-"*79)
	print("| {:<25}|{:^24}|{:^24} |".format("", "JOE BIDEN", "DONALD TRUMP"))
	print("-"*79)
	print("| {:<25}|{:>24}|{:>24} |".format("FRASI", nFrasiJB, nFrasiDT))
	print("| {:<25}|{:>24}|{:>24} |".format("TOKENS", JBLen, DTLen))
	print("-"*79)
	
	#calcolo e stampo la lunghezza media delle frasi e dei tokens
	print("\n2) la lunghezza media delle frasi in termini di token e delle parole in termini di caratteri\n")

	print("-"*79)
	print("| {:^75} |".format("LUNGHEZZA FRASI/TOKENS"))
	print("-"*79)
	print("| {:<25}|{:^24}|{:^24} |".format("", "JOE BIDEN", "DONALD TRUMP"))
	print("-"*79)
	print("| {:<25}|{:>24}|{:>24} |".format("TOKENS PER FRASE", JBLen / nFrasiJB, DTLen / nFrasiDT))
	print("| {:<25}|{:>24}|{:>24} |".format("CARATTERI PER TOKEN", mediaCaratteri(tokensListJB, JBLen), mediaCaratteri(tokensListDT, DTLen)))
	print("-"*79)
	
	#stampo la lunghezza del vocabolario e la ricchezza lessicale dei due corpus
	print("\n3) la grandezza del vocabolario e la ricchezza lessicale calcolata attraverso la Type Token Ratio (TTR), in entrambi i casi calcolati nei primi 5000 token\n")
	
	print("-"*79)
	print("| {:^75} |".format("LUNGHEZZA VOCABOLARIO - RICCHEZZA LESSICALE"))
	print("-"*79)
	print("| {:<25}|{:^24}|{:^24} |".format("", "JOE BIDEN", "DONALD TRUMP"))
	print("-"*79)
	print("| {:<25}|{:>24}|{:>24} |".format("PAROLE TIPO", len(set(tokensListJB[:4999])), len(set(tokensListDT[:4999]))))
	print("| {:<25}|{:>24}|{:>24} |".format("RICCHEZZA LESSICALE", (len(set(tokensListJB[:4999]))/5000), (len(set(tokensListDT[:4999]))/5000)))
	print("-"*79)
	
	#stampo le classi di frequenza V1, V5, e V10 ogni 500 tokens per ogni testo
	print("\n4) la distribuzione delle classi di frequenza |V1|, |V5| e |V10| all'aumentare del corpus per porzioni incrementali di 500 token (500 token, 1000 token, 1500 token, etc.)\n")
	
	print("-"*44)
	print("| {:^40} |".format("CORPUS DI JOE BIDEN"))
	print("-"*44)
	print("| {:^10}{:^10}{:^10}{:^10} |".format("TOKENS", "V1", "V5", "V10"))
	print("-"*44)
	analisiClassiFrequenza(tokensListJB)
	print("-"*44)
	
	print("\n" + "-"*44)
	print("| {:^40} |".format("CORPUS DI DONALD TRUMP"))
	print("-"*44)
	print("| {:^10}{:^10}{:^10}{:^10} |".format("TOKENS", "V1", "V5", "V10"))
	print("-"*44)
	analisiClassiFrequenza(tokensListDT)
	print("-"*44)
	
	#memorizzo i valori della funzione di analisi del PoSTagging
	mediaSJB, mediaVJB, densLexJB = analisiPoS(JBPoS, nFrasiJB)
	mediaSDT, mediaVDT, densLexDT = analisiPoS(DTPoS, nFrasiDT)
	
	#stampo la media dei sostantivi e dei verbi per frase per ogni corpus
	print("\n5) media di Sostantivi e Verbi per frase\n")
	
	print("-"*79)
	print("| {:^75} |".format("MEDIA SOSTANTIVI E VERBI"))
	print("-"*79)
	print("| {:<25}|{:^24}|{:^24} |".format("", "JOE BIDEN", "DONALD TRUMP"))
	print("-"*79)
	print("| {:<25}|{:>24}|{:>24} |".format("SOSTANTIVI PER FRASE", mediaSJB, mediaSDT))
	print("| {:<25}|{:>24}|{:>24} |".format("VERBI PER FRASE", mediaVJB, mediaVDT))
	print("-"*79)
	
	#stampo la densità lessicale
	print("\n6) la densità lessicale, calcolata come il rapporto tra il numero totale di occorrenze nel testo di Sostantivi, Verbi, Avverbi, Aggettivi e il numero totale di parole nel testo (ad esclusione dei segni di punteggiatura marcati con PoS \",\" \".\"): (|Sostantivi|+|Verbi|+|Avverbi|+|Aggettivi|)/(TOT-( |.|+|,| ) ).\n")
	
	print("-"*79)
	print("| {:^75} |".format("DENSITÀ LESSICALE"))
	print("-"*79)
	print("| {:<25}|{:^24}|{:^24} |".format("", "JOE BIDEN", "DONALD TRUMP"))
	print("-"*79)
	print("| {:<25}|{:>24}|{:>24} |".format("DENSITÀ LESSICALE", densLexJB, densLexDT))
	print("-"*79)
	
main(sys.argv[1], sys.argv[2])
