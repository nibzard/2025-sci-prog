# Customer Churn Prediction Project

## Sažetak projekta

Ovaj projekt analizira podatke telekomunikacijskih korisnika i razvija modele strojnog učenja za predviđanje churn-a (napuštanja usluge).

Projekt je obuhvaćao sljedeće korake:
1. Učitavanje i opis podataka  
2. Čišćenje i pripremu podataka  
3. Eksploratornu analizu podataka (EDA)  
4. Izgradnju i usporedbu modela strojnog učenja  
5. Evaluaciju performansi modela  

Cilj projekta bio je predvidjeti hoće li korisnik napustiti uslugu na temelju dostupnih atributa.

---

## Dataset

Korišten je Telco Customer Churn dataset.

Datoteke u projektu:
- Telco_customer_churn.xlsx – originalni dataset (Excel)
- Telco_customer_clean.csv – očišćeni dataset

Dataset sadrži demografske podatke, podatke o uslugama, ugovorima, naplati i ciljnu varijablu Churn.

---

## Struktura projekta

00_env_check.ipynb  
01_load_data.ipynb  
02_data_cleaning.ipynb  
03_EDA.ipynb  
04_train_and_compare_models.ipynb  
05_final_story.ipynb  

Telco_customer_churn.xlsx  
Telco_customer_clean.csv  

---

## Rezultati

Analiza je pokazala da oko 26.5% korisnika napušta uslugu, dok 73.5% ostaje, što ukazuje na umjerenu neuravnoteženost klasa.

Najvažniji faktori koji utječu na churn:
- trajanje korištenja usluge (Tenure Months)
- vrsta ugovora (Contract)
- mjesečni trošak (Monthly Charges)
- vrsta internetske usluge
- dostupnost tehničke podrške

Korisnici s month-to-month ugovorima i višim mjesečnim troškovima imaju veću vjerojatnost napuštanja usluge.

---

## Modeli

Izgrađena su dva modela:
- Logistic Regression
- Random Forest Classifier

Modeli su evaluirani pomoću accuracy, precision, recall, F1-score i ROC-AUC.

Logistic Regression je imao bolji ROC-AUC i recall za churn korisnike, dok je Random Forest imao veću ukupnu točnost.  
Zbog važnosti identifikacije churn korisnika, Logistic Regression je odabran kao preferirani model.

---

## Rasprava

Financijski i ugovorni faktori imaju najveći utjecaj na churn, dok demografski podaci imaju manji utjecaj.  
Modeli pokazuju potencijal za primjenu u stvarnom poslovnom okruženju.

