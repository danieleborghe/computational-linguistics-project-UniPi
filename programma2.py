import sys
import nltk
import math

def parser(frasi):
	tokensTot = []
	tokensPoSTot = []
	frasiTokenizzate = []
	
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		tokensPoS = nltk.pos_tag(tokens)
		
		tokensTot += tokens
		tokensPoSTot += tokensPoS
		
		frasiTokenizzate.append(tokens)
		
	return tokensTot, len(tokensTot), tokensPoSTot, frasiTokenizzate
	
def distribuzioneFrequenzaPoS(tokensPoSTot):
	#creo 5 liste vuote, una per ogni punto del progetto
	PoSList = []
	NNList = []
	VBList = []
	bigList1 = []
	bigList2 = []
	
	#inizializzo l'indice del ciclo while e creo una variabile bigramma (big)
	i = 0
	big = (tokensPoSTot[i-1], tokensPoSTot[i])
	
	#nel ciclo while utilizzo i bigrammi, e per questo parto dall'analisi del secondo elemento della lista di tokens
	#quindi ho creato questa if prima del while per analizzare anche il primo elemento della lista, che sarebbe rimasto escluso
	if big[1][1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
		VBList.append(big[1][0])
	elif big[1][1] in ["NN", "NNS", "NNP", "NNPS"]:
		NNList.append(big[1][0])
	
	while i < len(tokensPoSTot)-1:
		#incremento il contatore all'inizio del while per evitare di icrementarlo una volta prima del ciclo
		i += 1
		#ad ogni iterazione faccio scorrere i tokens del mio bigramma
		big = (tokensPoSTot[i-1], tokensPoSTot[i])
		
		#come prima operazione appendo alla lista di PoS i vari PoS tag che trovo lungo le iterazioni
		PoSList.append(big[1][1])
		
		#verifico con una doppia if che il token trovato è un verbo o un sostantivo
		if big[1][1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
			#se è un verbo lo appendo all'apPoSita lista
			VBList.append(big[1][0])
			
			#in questa if, verifico anche che il token precedente sia un sostantivo, e in caso lo aggiungo alla lista apPoSita
			if big[0][1] in ["NN", "NNS", "NNP", "NNPS"]:
				#in questo caso non ho bisogno di appendere il token anche a NNList, dato che sarà già stato appeso nell'iterazione precedente
				bigList1.append((big[0][0], big[1][0]))
		elif big[1][1] in ["NN", "NNS", "NNP", "NNPS"]:
			#se è un sostantivo lo appendo all'apPoSita lista
			NNList.append(big[1][0])
			
			#verifico che il token precedente sia un aggettivo, e in caso lo appendo alla lista apPoSita
			if big[0][1] in ["JJ", "JJR", "JJS"]:
				bigList2.append((big[0][0], big[1][0]))	
	
	#creo le distribuzioni di frequenza per ogni lista che ho creato
	PoSDist = nltk.FreqDist(PoSList)
	NNDist = nltk.FreqDist(NNList)
	VBDist = nltk.FreqDist(VBList)
	bigDist1 = nltk.FreqDist(bigList1)
	bigDist2 = nltk.FreqDist(bigList2)
	
	#ordino le distribuzioni di frequenza che ho ottenuto, in base ai numeri richiesti dal progetto
	PoSOrd = PoSDist.most_common(10)
	NNOrd = NNDist.most_common(20)
	VBOrd = VBDist.most_common(20)
	bigOrd1 = bigDist1.most_common(20)
	bigOrd2 = bigDist2.most_common(20)
	
	#stampo tutte le liste ottenute
	print("\n1.1) le 10 PoS (Part-of-Speech) più frequenti")
	print("-"*23)
	print("| {:^6}|{:^12} |".format("PoS", "FREQUENZA"))
	print("-"*23)
	for PoS in PoSOrd: print("| {:<6}|{:>12} |".format(PoS[0], PoS[1]))
	print("-"*23)
	
	print("\n1.2.1) i 20 sostantivi più frequenti")
	print("-"*32)
	print("| {:^15}|{:^12} |".format("TOKEN", "FREQUENZA"))
	print("-"*32)
	for NN in NNOrd: print("| {:<15}|{:>12} |".format(NN[0], NN[1]))
	print("-"*32)
	
	print("\n1.2.2) i 20 verbi più frequenti")
	print("-"*32)
	print("| {:^15}|{:^12} |".format("TOKEN", "FREQUENZA"))
	print("-"*32)
	for VB in VBOrd: print("| {:<15}|{:>12} |".format(VB[0], VB[1]))
	print("-"*32)

	print("\n1.3) i 20 bigrammi composti da un Sostantivo seguito da un Verbo più frequenti")
	print("-"*48)
	print("| {:^15}|{:^15}|{:^12} |".format("TOKEN 1", "TOKEN 2", "FREQUENZA"))
	print("-"*48)
	for big in bigOrd1: print("| {:<15}|{:<15}|{:>12} |".format(big[0][0], big[0][1], big[1]))
	print("-"*48)
	
	print("\n1.4) i 20 bigrammi composti da un Aggettivo seguito da un Sostantivo più frequenti")
	print("-"*48)
	print("| {:^15}|{:^15}|{:^12} |".format("TOKEN 1", "TOKEN 2", "FREQUENZA"))
	print("-"*48)
	for big in bigOrd2: print("| {:<15}|{:<15}|{:>12} |".format(big[0][0], big[0][1], big[1]))
	print("-"*48)
	
def analisiBigrammi(tokList, bigList, txtLen, bigDist, tokDist):
	#creo tre vocabolari, rispettivamente per la prob. congiunta, prob. condizionata e per la LMI
	probCgList = {}
	probCdList = {}
	LMIList = {}
	
	#calcolo il numero di bigrammi
	bigLen = len(bigList)
	
	#creo un ciclo for che scorra tutti i bigrammi
	for big in bigDist:
		#verifico che i token del bigramma abbiano entrambi frequenza maggiore di 3, grazie alla distribuzione passata alla funzione
		if tokDist[big[0]] > 3 and tokDist[big[1]] > 3:
			#calcolo la probabilità congiunta e la aggiungo al vocabolario
			probCg = bigDist[big] / bigLen
			probCgList[big] = probCg
		
			#calcolo la probabilità condizionata e la aggiungo al vocabolario
			probCd = bigDist[big] / tokDist[big[0]]
			probCdList[big] = probCd
			
			#calcolo la Local Mutual Information e la aggiuno al vocabolario
			LMI = bigDist[big] * math.log((bigDist[big] * bigLen) / (tokDist[big[0]] * tokDist[big[1]]), 2)
			LMIList[big] = LMI
	
	#ordino in ordine decrescente tutte e 3 le liste ottenute
	probCgOrd = sorted(probCgList.items(), key = lambda x: x[1], reverse = True)
	probCdOrd = sorted(probCdList.items(), key = lambda x: x[1], reverse = True)
	LMIOrd = sorted(LMIList.items(), key = lambda x: x[1], reverse = True)
	
	#eseguo la stampa delle tabelle
	print("\n2.1) con probabilità congiunta massima, indicando anche la relativa probabilità")
	print("\n" + "-"*79)
	print("| {:<15}{:<15}{:<30}{:<15} |".format("TOKEN 1", "TOKEN 2", "PROBABILITÀ CONGIUNTA", "PERCENTUALE"))
	print("-"*79)
	for p in probCgOrd[:20]: print("| {:<15}{:<15}{:<30}{:<15} |".format(p[0][0], p[0][1], p[1], round(p[1]*100, 2)))
	print("-"*79)
	
	print("\n2.2) con probabilità condizionata massima, indicando anche la relativa probabilità")
	print("\n" + "-"*79)
	print("| {:<15}{:<15}{:<30}{:<15} |".format("TOKEN 1", "TOKEN 2", "PROBABILITÀ CONDIZIONATA", "PERCENTUALE"))
	print("-"*79)
	for p in probCdOrd[:20]: print("| {:<15}{:<15}{:<30}{:<15} |".format(p[0][0], p[0][1], p[1], round(p[1]*100, 2)))
	print("-"*79)
		
	print("\n2.3) con forza associativa (calcolata in termini di Local Mutual Information) massima, indicando anche la relativa forza associativa")
	print("\n" + "-"*79)
	print("| {:<15}{:<15}{:<30}{:<15} |".format("TOKEN 1", "TOKEN 2", "L. MUTUAL INFORMATION", "ARROTONDAMENTO"))
	print("-"*79)
	for i in LMIOrd[:20]: print("| {:<15}{:<15}{:<30}{:<15} |".format(i[0][0], i[0][1], i[1], round(i[1], 2)))
	print("-"*79)
	
def calcoloMarkovUno(frasiTxt, frasi, tokList, txtLen, vocLen, bigDist, tokDist):
	#creo un vocabolario con 8 elementi (da 8 a 15), per ogni lunghezza di frase richiesta dal progetto
	probMax = {lenF: ['', 0.0] for lenF in range(8, 16)}
	
	#creo un ciclo for che scorra contemporaneamente le liste di tokens di ogni frase, e le frasi intese come stringhe
	for frase, txt in zip(frasi, frasiTxt):
		#calcolo la lunghezza della frase ad ogni iterazione
		lenF = len(frase)
	
		#verifico che la lunghezza della mia frase sia compresa tra 8 e 15
		if 8 <= lenF <= 15:
			#inizializzo il modello di markov 1 con la probabilità del primo token
			M1 = (tokDist[frase[0]] + 1) / (txtLen + vocLen)
			i = 0
			
			#creo un ciclo while che scorra i bigrammi nella lista di tokens della frase
			while i < lenF-1:
				#creo il mio bigramma che scorre ad ogni iterazione
				big = (frase[i], frase[i+1])
				#calcolo il modello di Markov 1 ad ogni iterazione
				M1 *= ((bigDist[big] + 1) / (tokDist[frase[i]] + vocLen))
				
				i += 1
			
			#verfico che il modello di markov ottenuto sia maggiore di quello già registrato nel vocabolario, per quella specifica lunghezza di frase
			if M1 > probMax[lenF][1]:
				#se il modello è maggiore del precedente allora lo sostituisco, inserendo a probMax[lenF][0] il testo della frase (txt)
				probMax[lenF] = [txt, M1]
	
	#stampo i modelli delle varie frasi
	print("-"*109)
	print("| {:<10}{:^70}{:<25} |".format("LUNGHEZZA", "FRASE", "MODELLO MARKOV 1"))
	print("-"*109)
	for m in probMax: print("| {:<10}{:<70}{:<25} |".format(m, probMax[m][0], probMax[m][1]))
	print("-"*109)
	
def analisiEntitaNominate(tokensPoS):
	tokensNET = nltk.ne_chunk(tokensPoS)
	PERSONList = []
	GPEList = []
	NEDic = {}
	
	for nodo in tokensNET:
		NE = ''
		
		#verifico che la classe nodo abbia l'attributo label
		if hasattr(nodo, 'label'):
		    	#verifico che il nodo analizzato sia intermedio
			if nodo.label() == "PERSON":
		        	#scorro tutte le foglie del nodo in analisi (le parti del nome)
				for foglia in nodo.leaves():
					NE = NE + ' ' + foglia[0]
					#se il nodo è una persona lo aggiungo alla lista delle persone
					PERSONList.append(NE)
			elif nodo.label() == "GPE":
				for foglia in nodo.leaves():
					NE = NE + ' ' + foglia[0]
					#se il nodo è un'entità geopolitica lo aggiungo alla lista GPE
					GPEList.append(NE)
	
	#creo il vocabolario della lista ottenuta
	NEVoc = set(PERSONList)
	#scorro il vocabolario e creo un dizionario con le frequenze di ogni entità
	for ent in NEVoc: NEDic[ent] = PERSONList.count(ent)
	#ordino il dizionario in ordine decrescente
	NEOrd = sorted(NEDic.items(), key = lambda x: x[1], reverse = True)
	
	#stampo i 15 nomi propri di persona più frequenti
	print("\n4.1) i 15 nomi propri di persona più frequenti (tipi), ordinati per frequenza")
	print("-"*37)
	print("| {:^20}|{:^12} |".format("ENTITÀ", "FREQUENZA"))
	print("-"*37)
	for ent in NEOrd[:15]: print("| {:<20}|{:>12} |".format(ent[0], ent[1]))
	print("-"*37)
	
	#svuoto il dizionario precedentemente creato
	NEDic.clear()
	
	#eseguo le stesse operazioni per le entità geo-politche
	NEVoc = set(GPEList)
	for ent in NEVoc: NEDic[ent] = GPEList.count(ent)
	NEOrd = sorted(NEDic.items(), key = lambda x: x[1], reverse = True)
	
	print("\n4.2) i 15 nomi propri di luogo più frequenti (tipi), ordinati per frequenza")
	print("-"*37)
	print("| {:^20}|{:^12} |".format("ENTITÀ", "FREQUENZA"))
	print("-"*37)
	for ent in NEOrd[:15]: print("| {:<20}|{:>12} |".format(ent[0], ent[1]))
	print("-"*37)
        
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
	
	#tokenizzo il testo e calcolo il numero di token con l'apPoSita funzione
	tokListJB, JBLen, JBPoS, frasiTokJB = parser(frasiJB)
	tokListDT, DTLen, DTPoS, frasiTokDT = parser(frasiDT)
	
	#estraggo i vocabolari
	tokVocJB = set(tokListJB)
	tokVocDT = set(tokListDT)
	
	#calcolo le distribuzioni di frequenza dei token
	tokDistJB = nltk.FreqDist(tokListJB)
	tokDistDT = nltk.FreqDist(tokListDT)
	
	#estraggo i bigrammi
	bigListJB = list(nltk.bigrams(tokListJB))
	bigListDT = list(nltk.bigrams(tokListDT))
	
	#calcolo le distribuzioni di frequenza dei bigrammi
	bigDistJB = nltk.FreqDist(bigListJB)
	bigDistDT = nltk.FreqDist(bigListDT)
	
	#stampo la lista PoS
	print("\n{:=^150}".format(" ANALISI DEL PART OF SPEECH TAGGING DEL TESTO DI JOE BIDEN "))
	print("\n1) estraete ed ordinate in ordine di frequenza decrescente, indicando anche la relativa frequenza:")
	distribuzioneFrequenzaPoS(JBPoS)
	
	print("\n{:=^150}".format(" ANALISI DEL PART OF SPEECH TAGGING DEL TESTO DI DONALD TRUMP "))
	print("\n1) estraete ed ordinate in ordine di frequenza decrescente, indicando anche la relativa frequenza:")
	distribuzioneFrequenzaPoS(DTPoS)
	
	#stampo l'analisi probabilistica dei testi
	print("\n{:=^150}".format(" ANALISI PROBABILISTICA DEL TESTO DI JOE BIDEN "))
	print("\n2) estraete ed ordinate i 20 bigrammi di token (dove ogni token deve avere una frequenzamaggiore di 3):")
	analisiBigrammi(tokListJB, bigListJB, JBLen, bigDistJB, tokDistJB)
	
	print("\n{:=^150}".format(" ANALISI PROBABILISTICA DEL TESTO DI DONALD TRUMP "))
	print("\n2) estraete ed ordinate i 20 bigrammi di token (dove ogni token deve avere una frequenzamaggiore di 3):")
	analisiBigrammi(tokListDT, bigListDT, DTLen, bigDistDT, tokDistDT)
	
	#stampo i modelli di markov dei testi
	print("\n{:=^150}".format(" MODELLI MASSIMI DI MARKOV 1 PER LE FRASI DI LUNGHEZZA DA 8 A 15 TOKENS NEL TESTO DI JOE BIDEN "))
	print("\n3) per ogni lunghezza di frase da 8 a 15 token, estraete la frase con probabilità più alta, dove la probabilità deve essere calcolata attraverso un modello di Markov di ordine 1 usando lo Add-one Smoothing. Il modello deve usare le statistiche estratte dal corpus che contiene le frasi")
	calcoloMarkovUno(frasiJB, frasiTokJB, tokListJB, JBLen, len(tokVocJB), bigDistJB, tokDistJB)
	
	print("\n{:=^150}".format(" MODELLI MASSIMI DI MARKOV 1 PER LE FRASI DI LUNGHEZZA DA 8 A 15 TOKENS NEL TESTO DI DONALD TRUMP "))
	print("\n3) per ogni lunghezza di frase da 8 a 15 token, estraete la frase con probabilità più alta, dove la probabilità deve essere calcolata attraverso un modello di Markov di ordine 1 usando lo Add-one Smoothing. Il modello deve usare le statistiche estratte dal corpus che contiene le frasi")
	calcoloMarkovUno(frasiDT, frasiTokDT, tokListDT, DTLen, len(tokVocDT), bigDistDT, tokDistDT)
	
	#stampo le entità nominate più frequenti
	print("\n{:=^150}".format(" ANALISI DELLE ENTITÀ NOMINATE NEL TESTO DI JOE BIDEN "))
	print("\n4) dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraete")
	analisiEntitaNominate(JBPoS)
	
	print("\n{:=^150}".format(" ANALISI DELLE ENTITÀ NOMINATE NEL TESTO DI DONALD TRUMP "))
	print("\n4) dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraete")
	analisiEntitaNominate(DTPoS)
	
main(sys.argv[1], sys.argv[2])
