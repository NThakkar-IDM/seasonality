""" SIACalendar.py

This script is an attempt at an easily updatable sia calendar
for Gavi countries. Primary source is the WHO excel spreadsheet found here:
https://www.who.int/immunization/monitoring_surveillance/data/en/ """
import sys
import numpy as np
import pandas as pd

## Country list for dataframe
## construction.
sys.path.append("..\\")
from utils.gavi_countries import countries

## Afghanistan campaigns
## Note, end date for the 2017 campaign was 
## imputed using the number of days from the previous campaign.
sia_calendar = [
("afghanistan","2009-04-21","2009-04-26","9-36M",1,"8 provinces"),
("afghanistan","2011-11-01","2011-12-31","9-59M",1,"9 Districts"),
("afghanistan","2011-04-01","2011-04-30","9-59M",1,"Various Provinces"),
("afghanistan","2011-11-26","2011-11-30","9M-10Y",1,"32 drought stricken districts"),
("afghanistan","2012-07-07","2012-07-12","9M-10Y",1,"in 17 provinces - 1st phase"),
("afghanistan","2012-12-01","2012-12-06","9M-10Y",1,"2nd phase"),
("afghanistan","2013-06-07","2013-09-30","9-59M",1,"new accessible areas - phase 1 & 2"),
("afghanistan","2014-06-01","2014-07-31","6M-10Y",1,"Flood-affected districts"),
("afghanistan","2014-08-01","2014-09-30","9-59M",1,"High-risk districts"),
("afghanistan","2015-05-01","2015-10-30","9-59M",1,"NaN"),
("afghanistan","2016-09-20","2016-09-25","9M-10Y",1,"NaN"),
("afghanistan","2017-04-01","2017-04-06","9-59M",1,"NaN"),
("afghanistan","2018-09-01","2018-11-26","9M-10Y",1,"Phase 1 started in 13/17 provinces"),
]

## Bangladesh campaigns
## Note, 2020 campaign has an imputed end date.
sia_calendar += [
("bangladesh","2010-02-14","2010-02-28","9-59M",1,"NaN"),
("bangladesh","2014-01-25","2014-02-13","9M-15Y",1,"Campaigns in educational institutes and community"),
("bangladesh","2016-12-11","2016-12-18","9M-<5Y",1,"NaN"),
("bangladesh","2017-04-29","2017-06-08","9M-<5Y",1,"4 districts of Sylhet division and Cox’s Bazar district of Chittagong division"),
("bangladesh","2017-09-16","2017-12-05","6M-<15Y",1,"NaN"),
("bangladesh","2020-02-01","2020-03-01","9-59M",1,"NaN"),
]

## Burkina Faso campaigns
sia_calendar += [
("burkina faso","2009-06-13","2009-06-17","6M-14Y",1,"NaN"),
("burkina faso","2011-04-08","2011-04-14","9-59M",1,"NaN"),
("burkina faso","2012-06-08","2012-06-12","6M+",1,"refugees"),
("burkina faso","2012-07-06","2012-07-09","6M+",1,"NaN"),
("burkina faso","2014-11-21","2014-11-30","9M-14Y",1,"NaN"),
("burkina faso","2018-07-27","2018-08-02","6-59M",1,"NaN"),
("burkina faso","2019-11-22","2019-11-28","9-59M",1,"NaN"),
]

## Burundi campaigns
## Note the 2011 campaign was dropped since it had no dates
sia_calendar += [
("burundi","2009-06-22","2009-06-26"," 6-59M",1,"NaN"),
("burundi","2010-10-25","2010-10-29","6M-14Y",1,"Bujumbura Mairie and Rural"),
("burundi","2010-06-07","2010-06-09"," 6-59M",1,"NaN"),
("burundi","2012-06-18","2012-06-21"," 6-59M",1,"NaN"),
("burundi","2013-06-24","2013-06-27","18-23M",1,"NaN"),
("burundi","2013-12-16","2013-12-19","18-23M",1,"NaN"),
("burundi","2014-05-06","2014-05-08","18-23M",1,"Semaine Africaine de Vaccination"),
("burundi","2014-12-16","2014-12-19","18-23M",1,"NaN"),
("burundi","2015-11-16","2015-11-19","18-23M",1,"Semaine Santé Mère Enfant 1er passage - 2nd dose"),
("burundi","2015-04-13","2015-04-16","18-23M",1,"Vaccination week"),
("burundi","2016-06-20","2016-06-23","18-23M",1,"2eme dose - Enfants de 18 à 23 mois non encore vaccinés au VAR2"),
("burundi","2016-12-06","2016-12-09","18-23M",1,"NaN"),
("burundi","2017-05-15","2017-05-19","9M-14Y",1,"Semaine Santé Mère Enfant 1er passage campaign report pending"),
("burundi","2017-12-05","2017-12-08","18-23M",1,"Enfants de 18 à 23 mois non encore vaccinés au RR2"),
("burundi","2018-06-19","2018-06-22","18-23M",1,"NaN"),
("burundi","2018-11-20","2018-11-23","18-23M",1,"Semaine santé Mère Enfant 2ème édition"),
]

## Cambodia campaigns
## Note, the 2017 campaign was split into 2 entries, with target pop based on
## an assumption that 15/25 provinces is 60% of the pop, etc.
## Also, 2020 campaign was removed since dates were not included.
sia_calendar += [
("cambodia","2011-11-01","2011-11-30","5-9Y",1,"NaN"),
("cambodia","2011-02-10","2011-03-31","9-59M",1,"NaN"),
("cambodia","2013-10-21","2013-12-26","9M-14Y",1,"NaN"),
("cambodia","2016-03-01","2016-03-31","9-59M",1,"NaN"),
("cambodia","2017-03-27","2017-04-03","6-59M",0.6,"Phase 1: 27 March – 3 April 2017 (15 provinces)"),
("cambodia","2017-05-02","2017-05-09","6-59M",0.4,"Phase 2: 2-9 May 2017 (10 provinces)"),
]

## Cameroon campaigns
## The 2012 campaign was split into 2 entries for north and south regions,
## with dates adjusted based on the notes.
sia_calendar += [
("cameroon","2009-06-30","2009-07-05","9-59M",1,"NaN"),
("cameroon","2010-03-03","2010-03-07","9-59M",1,"NaN"),
("cameroon","2010-05-01","2010-06-03","9-59M",1,"NaN"),
("cameroon","2011-04-01","NaT","9-59M",1,"NaN"),
("cameroon","2012-03-27","NaT","9-59M",0.3,"27 March in 3 N. Regions"),
("cameroon","2012-04-23","NaT","9-59M",0.7,"23 April 2012 in 7 S Regions"),
("cameroon","2015-11-24","2015-11-29","9M-14Y",1,"NaN"),
("cameroon","2015-08-01","NaT","9M+",1,"NaN"),
("cameroon","2018-05-09","2018-05-12","9-59M",1,"and 4-8/06/2018"),
("cameroon","2019-12-04","2019-12-08","9-59M",1,"NaN"),
]

## Chad campaigns
## The 2020 campaign was removed since it had no dates.
## The 2012 campaign was split into two entries.
## The 2016 note has much more information, not included here.
## The 2018 note has much more information, also not included here.
'''
sia_calendar += [
("chad","2009-03-05","2009-03-10","9-59M",1,"NaN"),
("chad","2009-03-14","2009-03-19","9-59M",1,"NaN"),
("chad","2009-01-26","2009-01-31","9-59M",1,"NaN"),
("chad","2011-03-01","2011-05-01","unknown",1,"NaN"),
("chad","2012-01-20","2012-01-26","6-59M",13./22.,"SIAs in 2 phases: 13 Regions 20-26 January."),
("chad","2012-01-26","2012-02-02","6-59M",9./22.,"SIAs in 2 phases: 9 Regions SIAs 27 Jan. to 2 Feb 2012"),
("chad","2014-10-20","2014-10-26","6M-9Y",1,"2nd phase - 38 districts"),
("chad","2014-06-16","2014-06-22","6M-9Y",1,"1st phase"),
("chad","2016-11-21","2016-11-27","9-59M",1,"Phase 1: 14 régions..."),
("chad","2016-05-25","2016-05-31","9-59M",1,"NaN"),
("chad","2017-03-12","2017-03-18","9-59M",1,"Phase 2: 9 regions"),
("chad","2018-11-17","2018-11-23","6M-9Y",1,"Baher El Gazel, Borkou, Ennedi Est, Ennedi Oue..."),
("chad","2019-04-01","NaT","6M-9Y",1,"or 6M-9Y - TP 6,265,912"),
]
'''
sia_calendar += [
("chad","2009-01-26","2009-01-31","9-59M",75380./(75380+1309694+365074),"NaN"),
("chad","2009-03-05","2009-03-10","9-59M",1309694./(75380+1309694+365074),"NaN"),
("chad","2009-03-14","2009-03-19","9-59M",365074./(75380+1309694+365074),"NaN"),
("chad","2012-01-20","2012-01-26","6-59M",13./22.,"SIAs in 2 phases: 13 Regions 20-26 January."),
("chad","2012-01-26","2012-02-02","6-59M",9./22.,"SIAs in 2 phases: 9 Regions SIAs 27 Jan. to 2 Feb 2012"),
("chad","2014-10-20","2014-10-26","6M-9Y",2349620./(2349620+2549188),"2nd phase - 38 districts"),
("chad","2014-06-16","2014-06-22","6M-9Y",2549188./(2349620+2549188),"1st phase"),
("chad","2016-05-25","2016-05-31","9M-14Y",414392./(414392+2342341+707103),"conducted in 7 districts"),
("chad","2016-11-21","2016-11-27","9-59M",2342341./(414392+2342341+707103),"Phase 1: 14 régions : Barh El Ghazal, Batha, C..."),
("chad","2017-03-12","2017-03-18","9-59M",707103./(414392+2342341+707103),"Phase 2: 9 regions"),
("chad","2018-11-17","2018-11-23","9M-9Y",2421067./(2421067+653535+106965+95198+103543+467455),"Baher El Gazel, Borkou, Ennedi Est, Ennedi Oue..."),
("chad","2019-04-09","2019-04-15","6M-9Y",653535./(2421067+653535+106965+95198+103543+467455),"NaN"),
("chad","2019-01-01","2019-02-28","6M-9Y",106965./(2421067+653535+106965+95198+103543+467455),"NaN"),
("chad","2019-06-01","2019-06-30","6M-9Y",95198./(2421067+653535+106965+95198+103543+467455),"NaN"),
("chad","2019-06-26","2019-07-31","6M-9Y",103543./(2421067+653535+106965+95198+103543+467455),"NaN"),
("chad","2019-12-12","2019-12-18","9M-9Y",467455./(2421067+653535+106965+95198+103543+467455),"NaN"),
("chad","2021-01-12","2021-01-18","9-59M",1950968./(1950968+1745337),"Phase 1 of the national SIA with two phases in..."),
("chad","2021-03-22","2021-03-28","9-59M",1745337./(1950968+1745337),"Bloc 2"),
]

## Democratic people's republic of korea campaign
sia_calendar += [
("democratic people's republic of korea","2019-10-01","NaT","9M-14Y",1,"NaN"),
]

## Democrate republic of the congo campaigns
## Entries here are very detailed, with individual entries for individual phases of
## the campaign. There's some redundancy in the dates as a result (i.e. multiple entries
## with the same start and end date). These are combined in processing.
sia_calendar += [
("democratic republic of the congo","2009-11-10","2009-11-14","6-59M",1,"NaN"),
("democratic republic of the congo","2010-01-20","2010-01-24","6-59M",1,"Bandundu"),
("democratic republic of the congo","2011-04-01","NaT","6-59M",1,"Kasai Occidental"),
("democratic republic of the congo","2011-02-01","NaT","6-59M",1,"Katanna"),
("democratic republic of the congo","2011-03-01","NaT","6-59M",1,"Sud Kivu"),
("democratic republic of the congo","2011-05-01","NaT","6-59M",1,"81 ZS Katanna"),
("democratic republic of the congo","2011-06-01","NaT","6-59M",1,"Kasai Oriental"),
("democratic republic of the congo","2012-01-24","2012-01-28","6-59M",1,"Nord Kivu"),
("democratic republic of the congo","2012-01-24","2012-01-28","6-59M",1,"Bas-Congo"),
("democratic republic of the congo","2012-01-24","2012-01-28","6M-14Y",1,"Bas Congo"),
("democratic republic of the congo","2012-03-07","2012-03-11","6-59M",1,"Kasai Oriental"),
("democratic republic of the congo","2012-09-11","2012-09-15","6-59M",1,"Equateur"),
("democratic republic of the congo","2012-10-04","2012-10-08","6M-14Y",1,"Bas-Congo"),
("democratic republic of the congo","2012-11-13","2012-11-20","6-59M",1,"Kasai Oriental"),
("democratic republic of the congo","2012-01-24","2012-01-28","6-59M",1,"Bandundu"),
("democratic republic of the congo","2012-08-22","2012-08-26","6-59M",1,"Nord Kivu"),
("democratic republic of the congo","2012-01-24","2012-01-28","6-59M",1,"Bandundu"),
("democratic republic of the congo","2012-01-24","2012-01-28","6M-14Y",1,"Nord Kivu"),
("democratic republic of the congo","2012-10-29","2012-11-03","6-59M",1,"Bandundu"),
("democratic republic of the congo","2012-08-22","2012-08-26","6-59M",1,"Katanga"),
("democratic republic of the congo","2012-08-22","2012-08-26","6-59M",1,"Maniema"),
("democratic republic of the congo","2012-08-21","2012-08-25","6-59M",1,"Kasai Occidental"),
("democratic republic of the congo","2013-09-24","2013-09-28","6M-9Y",1,"Equateur, Pr. Orientale"),
("democratic republic of the congo","2013-12-10","2013-12-14","6M-9Y",1,"Nord & Sud Kivu"),
("democratic republic of the congo","2013-01-01","2013-01-31","6M-14Y",1,"N. Kivu"),
("democratic republic of the congo","2014-03-18","2014-03-22","6M-9Y",1,"Phase 2: Katanga + Maniema\r\nTarget: 4469008 ..."),
("democratic republic of the congo","2014-05-27","2014-05-31","6M-9Y",1,"Phase 2: Kasai Occ., Kasai Oriental"),
("democratic republic of the congo","2014-06-24","2014-06-28","6M-9Y",1,"Kinshasa"),
("democratic republic of the congo","2014-07-29","2014-08-02","6M-9Y",1,"Bas Congo & Bandundu"),
("democratic republic of the congo","2014-08-05","2014-08-09","6M-9Y",1,"Bandundu"),
("democratic republic of the congo","2015-03-01","2015-04-30","6-59M",1,"Equateur, 3 ZS, 24 AS"),
("democratic republic of the congo","2015-07-01","2015-07-31","6-59M",1,"Katanga, 1 ZS (Nyunzu)"),
("democratic republic of the congo","2015-04-15","2015-04-19","6-59M",1,"Katanga, 1 ZS (Kabondo Dianda)"),
("democratic republic of the congo","2015-09-01","2015-09-05","6M-14Y",1,"Katange, 1 ZS (Nyemba)"),
("democratic republic of the congo","2015-07-01","2015-07-31","6M-15Y",1,"Maniema, 1 ZS (Kasongo)"),
("democratic republic of the congo","2015-12-15","2015-12-19","6M-15Y",1,"K Or"),
("democratic republic of the congo","2015-12-11","2015-12-15","6M-15Y",1,"N Kivu"),
("democratic republic of the congo","2015-12-15","2015-12-19","6M-15Y",1,"Katanga"),
("democratic republic of the congo","2015-03-16","2015-03-20","6-59M",1,"N. Kivu, 1 ZS (Mweso), 23 AS"),
("democratic republic of the congo","2015-03-04","2015-03-08","6-59M",1,"S. Kivu, 1 ZS (Kalehe), 16 AS"),
("democratic republic of the congo","2015-05-26","2015-06-03","6-59M",1,"S Kivu, 1 ZS (Bunyakiri)"),
("democratic republic of the congo","2015-05-01","2015-05-31","6-59M",1,"NaN"),
("democratic republic of the congo","2015-05-01","2015-05-31","6M-10Y",1,"Katanga, 1 ZS (Malemba Nkulu)"),
("democratic republic of the congo","2015-05-01","2015-06-30","6-59M",1,"Katanga, 1 ZS (Kilwa)"),
("democratic republic of the congo","2016-08-27","2016-08-31","6-59M",1,"Phase 1 completed\r\nPhase 2 and 3: 6 October ..."),
("democratic republic of the congo","2016-10-06","2016-10-10","6-59M",1,"Phase 2: Kinshasa, Kongo Central, Kwango, Kwil..."),
("democratic republic of the congo","2017-02-14","2017-02-18","6-59M",1,"Phase 3: Haut Katanga, Lualaba, Haut Lomami, T..."),
("democratic republic of the congo","2018-12-15","2018-12-31","6-59M",1,"NaN"),
("democratic republic of the congo","2019-01-01","2019-08-30","6-59M",1,"328 aires de santé in 28/35 zones de santé in ..."),
]

## Eritrea campaigns
sia_calendar += [
("eritrea","2009-05-06","2009-05-10","9-47M",1,"NaN"),
("eritrea","2012-04-25","2012-04-29","9-47M",1,"NaN"),
("eritrea","2015-04-22","2015-04-26","9-59M",1,"NaN"),
("eritrea","2018-11-21","2018-11-30","9M-14Y",1,"NaN"),
]

## Ethiopia calendar
## Note, 3 entries for the 2009 campaign were reduced to 1.
## 2020 campaign was removed.
sia_calendar += [
("ethiopia","2009-01-01","2009-01-31","6-59M",1,"NaN"),
("ethiopia","2009-06-10","2009-06-17","6-59M",1,"Afar Region"),
("ethiopia","2010-02-18","2010-04-30","6-59M",1,"Outbreak Response"),
("ethiopia","2010-10-22","2010-10-25","9-47M",1,"Except Afar Tigray and Gambella (2011)"),
("ethiopia","2011-02-18","2011-02-21","9-47M",1,"Targeting 4 Major Regions Tigray, Benshangul Gumz, Gambella and Afar regions)"),
("ethiopia","2011-10-01","2011-11-01","6M-14Y",1,"NaN"),
("ethiopia","2013-05-29","2013-06-05","9-59M",1,"National SIA"),
("ethiopia","2015-10-19","2016-01-31","6-59M",1,"including all districts"),
("ethiopia","2016-04-22","2016-04-28","6M-<15Y",1,"The campaign was conducted from April 22 – 28, 2016 in all other regions except April 26 – May 2nd in SNNPR. Integrated with OPV"),
("ethiopia","2017-02-23","2017-03-10","9M-14Y",1,"(9–59 months nationwide and 5 - <15 years in all woredas not covered in April 2016 SIA)"),
("ethiopia","2017-07-29","2017-08-04","6-179M",1,"Somali Region - final campaign report pending"),
]

## Gambia campaigns
sia_calendar += [
("gambia","2011-12-12","2011-12-18","9-59M",1,"NaN"),
("gambia","2016-04-25","2016-05-01","9M-14Y",1,"NaN"),
]

## Ghana campaigns
sia_calendar += [
("ghana","2010-11-02","2010-11-06","9-59M",1,"NaN"),
("ghana","2013-09-11","2013-09-20","9M-14Y",1,"NaN"),
("ghana","2018-10-17","2018-10-22","9-59M",1,"NaN"),
]

## India campaigns
## Entries here are very detailed, with individual entries for individual phases of
## the campaign. There's some redundancy in the dates as a result (i.e. multiple entries
## with the same start and end date). These are combined in processing.
sia_calendar += [
("india","2010-11-01","2011-01-31","9M-10Y",1,"PHASE I 45 districts in 13 states"),
("india","2011-01-31","2011-07-09","9M-10Y",1,"Phase 1"),
("india","2011-09-19","2012-03-27","9M-10Y",1,"Phase 2 - 14 States in the country"),
("india","2012-01-09","2012-06-06","9M-9Y",1,"Phase 2"),
("india","2012-09-11","2013-03-04","9M-9Y",1,"NaN"),
("india","2013-01-28","2013-11-26","9M-9Y",1,"NaN"),
("india","2015-12-17","NaT","9M-15Y",1,"NaN"),
("india","2017-02-07","2017-03-19","9M-15Y",1,"State: Karnataka - target pop TBC"),
("india","2017-08-17","2017-09-30","9M-15Y",1,"State: Telangana"),
("india","2017-08-30","2017-10-15","9M-15Y",1,"State: Himachal Pradesh"),
("india","2017-10-03","2018-03-30","9M-15Y",1,"State: Kerala"),
("india","2017-10-30","2018-01-20","9M-15Y",1,"State: Uttarakhand"),
("india","2017-02-06","2017-05-31","9M-15Y",1,"State: Tamil Nadu - target pop TBC"),
("india","2017-02-08","2017-03-20","9M-15Y",1,"State: Goa"),
("india","2017-02-06","2017-06-10","9M-15Y",1,"State: Puducherry"),
("india","2017-02-06","2017-05-30","9M-15Y",1,"State: Lakshadweep"),
("india","2017-08-01","2017-09-08","9M-15Y",1,"State: Andhra Pradesh"),
("india","2017-08-04","2017-10-31","9M-15Y",1,"State: Chandigarh"),
("india","2017-08-03","2017-09-11","9M-15Y",1,"State: Daman & Diu"),
("india","2017-08-02","2017-09-13","9M-15Y",1,"State: Dadra & Nagar Haveli"),
("india","2018-01-29","2018-04-06","9M-15Y",1,"State: Odisha,Arunachal Pradesh, Manipur, Odisha"),
("india","2018-07-26","2018-12-24","9M-15Y",1,"State: Jharkhand"),
("india","2018-08-18","2018-12-31","9M-15Y",1,"State: Assam"),
("india","2018-09-24","2018-12-15","9M-15Y",1,"State: Jammu & Kashmir"),
("india","2018-09-15","2019-01-05","9M-15Y",1,"State: Tripura"),
("india","2018-09-24","NaT","9M-15Y",1,"State: Meghalaya"),
("india","2018-10-06","NaT","9M-15Y",1,"State: Chhattisgarh"),
("india","2018-11-26","NaT","9M-15Y",1,"State: Uttar Pradesh"),
("india","2018-11-27","NaT","9M-15Y",1,"State: Maharashtra"),
("india","2018-02-01","2018-03-13","9M-15Y",1,"State: Arunachal Pradesh"),
("india","2018-03-26","2018-06-26","9M-15Y",1,"State: Manipur, Assam, Haryana, Mizoram,Punjab, A&N Islands"),
("india","2018-04-16","2018-05-28","9M-15Y",1,"State: Mizoram"),
("india","2018-04-12","2018-08-30","9M-15Y",1,"State: Andaman & Nicobar"),
("india","2018-05-01","2018-08-31","9M-15Y",1,"State: Punjab"),
("india","2018-07-16","2018-11-14","9M-15Y",1,"State: Gujarat"),
("india","2018-10-03","2018-12-08","9M-15Y",1,"State: Nagaland"),
("india","2018-04-25","2018-09-15","9M-15Y",1,"State: Haryana"),
("india","2019-01-15","NaT","9M-15Y",1,"State: Bihar - preliminary results"),
("india","2019-01-15","NaT","9M-15Y",1,"State: Madhya Pradesh - preliminary results"),
("india","2019-01-16","NaT","9M-15Y",1,"State: Delhi"),
("india","2019-07-01","NaT","9M-15Y",1,"State: Rajasthan"),
("india","2019-08-01","NaT","9M-15Y",1,"State: Sikkim"),
]

## Indonesia campaigns
## Entries for the 2009 campaign were consolidated into a single entry.
sia_calendar += [
("indonesia","2009-10-01","2009-10-31","9-59M",1,"4 provinces/in disaster areas/Papua"),
("indonesia","2010-10-05","2010-11-06","9-59M",1,"NaN"),
("indonesia","2011-10-18","2011-11-18","9-59M",1,"3rd Phase"),
("indonesia","2016-08-01","2016-08-14","9-59M",1,"183 districts in 28 provinces, integrated with VitA"),
("indonesia","2017-08-01","2017-09-30","9M-15Y",1,"1st phase: 6 districts in Java Island, 2nd phase in 2018 for rest of the country"),
("indonesia","2018-08-01","2018-12-31","9M-15Y",1,"2nd phase: All provinces except six provinces in Java Island (1st phase in 2017)"),
]

## Kenya campaigns
## Note, the 2020 campaign entry was removed.
sia_calendar += [
("kenya","2009-09-19","2009-09-25","9-59M",1,"NaN"),
("kenya","2012-11-01","2012-11-07","9-59M",1,"NaN"),
("kenya","2016-05-16","2016-05-24","9M-14Y",1,"NaN"),
("kenya","2018-10-27","2018-10-31","6-59M",1,"NaN"),
("kenya","2018-12-03","2018-12-07","6-59M",1,"NaN"),
]

## Lesotho campaigns
## Note, the 2020 campaign entry was dropped.
sia_calendar += [
("lesotho","2010-09-20","2010-10-01","6M-15Y",1,"NaN"),
("lesotho","2013-10-14","2013-10-25"," 9-59M",1,"NaN"),
("lesotho","2017-02-13","2017-02-24","9M-14Y",1,"NaN"),
]

## Malawi campaigns
## Notes for the 2017 entry were truncated.
sia_calendar += [
("malawi","2010-08-16","2010-08-20","9M-15Y",1,"in all 28 districts"),
("malawi","2013-11-02","2013-11-06"," 6-59M",1,"NaN"),
("malawi","2015-05-13","2015-05-15"," 9-59M",1,"NaN"),
("malawi","2017-06-12","2017-06-16","9M-14Y",1,"Integrated with VitA (6-59months) and deworming (12-59months)..."),
]

## Mozambique campaigns
## The 2018 entry was split into 2
sia_calendar += [
("mozambique","2011-05-23","2011-05-27","6-59M",1,"NaN"),
("mozambique","2013-12-02","2013-12-06","6-59M",1,"NaN"),
("mozambique","2018-04-09","2018-04-13","6M-15Y",0.5,"SIAs in the Northern Provinces + Zambézia - (1st phase): 9–13 April 2018"),
("mozambique","2018-05-21","2018-05-25","6M-15Y",0.5,"SIAs in the Central and South Provinces - (2nd phase): 21–25 May 2018"),
]

## Myanmar campaigns
sia_calendar += [
("myanmar","2012-03-22","2012-03-31","9-59M",1,"NaN"),
("myanmar","2015-01-19","2015-01-27","5-14Y",1,"Phase 1"),
("myanmar","2015-02-19","2015-02-28","9-59M",1,"Phase 2"),
("myanmar","2019-10-01","NaT","9-59M",1,"Round 2"),
("myanmar","2019-02-01","NaT","9M-15Y",1,"Yangon"),
]

## Nepal campaigns
## Note, deleted the 2020 entry
sia_calendar += [
("nepal","2012-02-26","2013-03-12","9M-14Y",1,"1st Phase"),
("nepal","2012-09-17","2012-10-16","9M-14Y",1,"2nd phase"),
("nepal","2012-12-14","2013-01-14","9M-14Y",1,"3rd phase"),
("nepal","2015-08-15","2015-09-15","6M-5Y",1,"14 districts"),
("nepal","2016-02-07","2016-04-12","9-59M",1,"4 phases\r\nintegrated with OPV"),
]

## Nigeria calendar
## Note, I deleted the 2020 entries.
sia_calendar += [
("nigeria","2011-01-26","2011-02-23","9-59M",1,"Northern States 26 to 30 Jan. Southern  States 23 to 27 Feb."),
("nigeria","2013-10-05","2013-10-09","9-59M",1,"Phase 1: 20 Northern states"),
("nigeria","2013-04-13","2013-04-16"," 6-59M",1,"NaN"),
("nigeria","2013-11-02","2013-11-06","9-59M",1,"Phase 2: 17 Southern states"),
("nigeria","2015-11-21","2015-11-25","6M-10Y",1,"Northern states"),
("nigeria","2016-01-28","2016-02-01","9-59M",1,"Phase 2 (Phase 1 in 2015): Southern states (17)..."),
("nigeria","2017-11-09","2018-04-30","9-59M",1,"NaN"),
("nigeria","2017-01-12","2017-02-06","6M-10Y",1,"OBR (Adamawa, Borno and Yobe states)"),
("nigeria","2018-03-01","2018-03-30","9-59M",1,"NaN"),
("nigeria","2019-10-31","2019-11-30","9-59M",1,"Kano State: 31/10-05/11 - Nasawara State: 30/11 - Kogi State: 09/12 - Rest of North: 16/11"),
]

## Pakistan campaigns
## Notes section containing more detailed age-target information for 2018 campaign was
## cut off.
sia_calendar += [
("pakistan","2010-09-20","2010-10-02","6-59M",1,"Mopup"),
("pakistan","2010-02-22","2010-03-05","9M-<13Y",1,"Flood response - Balochistan (16 Districts out of 30 Districts)"),
("pakistan","2010-10-25","2010-11-05","6-59M",1,"NaN"),
("pakistan","2011-01-05","2011-01-17","9-59M",1,"NaN"),
("pakistan","2011-01-07","2011-01-29","9-59M",1,"NaN"),
("pakistan","2011-01-10","2011-01-22","9-59M",1,"NaN"),
("pakistan","2011-07-04","2011-07-09","9-59M",1,"NaN"),
("pakistan","2011-11-21","2011-11-26","9-59M",1,"NaN"),
("pakistan","2012-12-24","2013-01-04","9M-9Y",1,"NaN"),
("pakistan","2013-01-01","2013-03-31","9M-9Y",1,"NaN"),
("pakistan","2013-04-29","2013-09-14","6M-9Y",1,"NaN"),
("pakistan","2014-05-19","2014-05-31","6M-9Y",1,"Sindh"),
("pakistan","2014-12-08","2014-12-20","6M-9Y",1,"AJK"),
("pakistan","2014-05-19","2014-05-31","6M-9Y",1,"KP"),
("pakistan","2015-01-26","2015-02-09","6M-10Y",1,"Punjab"),
("pakistan","2015-02-09","2015-02-21","6M-10Y",1,"CDA - Islamabad"),
("pakistan","2015-04-13","2015-04-25","6M-10Y",1,"Balochistan"),
("pakistan","2015-02-16","2015-02-28","6M-10Y",1,"ICT - Islamabad"),
("pakistan","2015-05-18","2015-05-30","6M-10Y",1,"Gilgit Baltistan"),
("pakistan","2015-08-20","2015-08-31","6M-10Y",1,"FATA"),
("pakistan","2017-08-01","NaT","9-59M",1,"NaN"),
("pakistan","2018-10-15","2018-10-27","9-59M",1,"NaN"),
("pakistan","2018-05-11","2018-05-17","9-119M",1,"High Risk UCs of 5 districts of Balochistan,..."),
("pakistan","2018-05-14","2018-06-07","6-59M",1,"181 selected UCs in 17 districts of Punjab"),
]

## Papua new guinea campaigns
## Start and end dates are VERY far apart here. Not sure why?
sia_calendar += [
("papua new guinea","2008-03-01","2009-06-30"," 6-83M",1,"NaN"),
("papua new guinea","2010-07-01","2010-09-01"," 6M-2Y",1,"Most provinces have not reported"),
("papua new guinea","2012-04-15","2012-06-25"," 6-35M",1,"NaN"),
("papua new guinea","2015-08-12","2015-12-31","9M-14Y",1,"NaN"),
("papua new guinea","2016-03-01","2016-05-31","9M-15Y",1,"NaN"),
("papua new guinea","2017-10-24","2018-01-31","6M-15Y",1,"NaN"),
("papua new guinea","2019-06-01","NaT"," 6-59M",1,"NaN"),
]

## Rwanda campaigns
sia_calendar += [
("rwanda","2009-10-06","2009-10-10","9-59M",1,"NaN"),
("rwanda","2013-03-12","2013-03-15","9M-14Y",1,"NaN"),
("rwanda","2017-10-09","2017-10-13","9-59M",1,"NaN"),
("rwanda","2017-03-13","2017-03-18","9-15Y",1,"NaN"),
]

## Sao tome and principe campaigns
sia_calendar += [
("sao tome and principe","2012-07-23","2012-07-27","9-59M",1,"NaN"),
("sao tome and principe","2016-12-05","2016-12-10","9M-14Y",1,"NaN"),
]

## Senegal campaigns
sia_calendar += [
("senegal","2010-11-19","2010-11-25","9-59M",1,"NaN"),
("senegal","2010-03-16","2010-03-20","6-59M",1,"NaN"),
("senegal","2013-11-18","2013-11-27","9M-14Y",1,"NaN"),
("senegal","2017-11-20","2017-11-26","9-59M",1,"NaN"),
]

## Sierra leone campaigns
sia_calendar += [
("sierra leone","2009-11-24","2009-11-29","9-59M",1,"NaN"),
("sierra leone","2012-05-24","2012-05-31","9-59M",1,"NaN"),
("sierra leone","2015-06-05","2015-06-10","9-59M",1,"integrated with Polio"),
("sierra leone","2015-11-26","2015-11-30","15-23M",1,"NaN"),
("sierra leone","2016-04-25","2016-05-01","6M-14Y",1,"NaN"),
("sierra leone","2018-07-03","2018-07-08","6M-15Y",1,"NaN"),
("sierra leone","2019-06-10","2019-06-16","9M-14Y",1,"NaN"),
("sierra leone","2019-01-19","2019-01-25","6M-15Y",1,"NaN"),
]

## Solomon islands campaigns
sia_calendar += [
("solomon islands","2009-09-21","2009-10-16","12-48M",1,"NaN"),
("solomon islands","2012-06-01","2012-08-01","12-59M",1,"NaN"),
("solomon islands","2014-09-01","2014-12-19","6M-29Y",1,"NaN"),
("solomon islands","2019-09-01","NaT","1-4Y",1,"NaN"),
]

## United republic of tanzania
sia_calendar += [
("united republic of tanzania","2011-11-12","2011-11-13","9-59M",1,"NaN"),
("united republic of tanzania","2014-10-18","2014-10-24","9M-14Y",1,"NaN"),
("united republic of tanzania","2019-10-15","2019-10-19","9M-5Y",1,"26-30 Sept: Zanzibar + 17-21 Oct: in mainland"),
]

## Viet Nam campaigns
## 2014 and 2016 campaigns were dropped since there were no dates.
## 2018 campaign (2nd entry) was dropped for the same reason.
sia_calendar += [
("viet nam","2010-09-10","2010-11-30","12-59M",1,"all 63 provinces"),
("viet nam","2013-04-01","2013-12-31","1-15Y",1,"NaN"),
("viet nam","2014-09-15","2015-05-15","1-14Y",1,"NaN"),
("viet nam","2018-09-01","NaT","1-4Y",1,"6 provinces"),
("viet nam","2018-09-01","NaT","1-5Y",1,"13 provinces"),
]

## Yemen calendar
## 2020 entry was removed.
sia_calendar += [
("yemen","2009-12-12","2009-12-17","9-5Y",1,"NaN"),
("yemen","2009-01-01","2009-01-31","9M-5Y",1,"NaN"),
("yemen","2010-01-01","2010-07-31","<15Y",1,"NaN"),
("yemen","2011-09-01","NaT","6M-15Y",1,"NaN"),
("yemen","2012-03-10","2012-03-15","6M-10Y",1,"NaN"),
("yemen","2013-06-01","2013-07-31","6M-10Y",1,"Sa'ada Governate"),
("yemen","2014-11-09","2014-11-20","9M-15Y",1,"SIA"),
("yemen","2015-08-15","2015-08-20","6M-15Y",1,"Sub National (Abayn, Al Baida, Al Jawf, Al Mahweet, Taiz, Hajjah, Shabwah, Sa’ada, Sana’a, Amran) governorates"),
("yemen","2016-01-01","NaT","6M-15Y",1,"NaN"),
("yemen","2017-03-01","NaT","6M-15Y",1,"Mop up in 18 districts"),
("yemen","2017-05-01","NaT","6M-15Y",1,"Mop up in 6 districts"),
("yemen","2019-02-09","2019-02-14","6M-14Y",1,"except Sa'adah"),
("yemen","2018-03-01","NaT","6M-10Y",1,"NaN"),
("yemen","2018-05-01","NaT","6M-15Y",1,"NaN"),
]

## Zambia campaigns
## 2020 entry was dropped.
sia_calendar += [
("zambia","2010-07-19","2010-07-24","9-47M",1,"NaN"),
("zambia","2012-09-10","2012-09-15","<15Y",1,"NaN"),
("zambia","2016-09-19","2016-09-24","9M-14Y",1,"NaN"),
]

## Zimbabwe campaigns
sia_calendar += [
("zimbabwe","2009-06-08","2009-06-12"," 9-59M",1,"NaN"),
("zimbabwe","2010-05-01","2010-05-31","6M-14Y",1,"NaN"),
("zimbabwe","2012-06-18","2012-06-22"," 6-59M",1,"NaN"),
("zimbabwe","2015-09-28","2015-10-02","9M-14Y",1,"NaN"),
("zimbabwe","2019-09-23","2019-09-27"," 9M-5Y",1,"NaN"),
]

## Somalia campaigns
sia_calendar += [
("somalia","2005-11-01","2005-12-01","9M-15Y",1,"NW, NE & 1 region CSZ: Somaliland, Bakool, Put..."),
("somalia","2006-05-01","2006-05-31","9M-15Y",1,"South and central"),
("somalia","2007-02-08","2007-09-25","9M-15Y",1,"Wanle Weyne, Afgoy, Awdegley, Mahas"),
("somalia","2008-03-01","NaT","9M-15Y",1,"NaN"),
("somalia","2008-12-01","NaT","9M-15Y",1,"NaN"),
("somalia","2009-01-01","2009-05-31","9-59M",1,"North East Zone & North West Zone"),
("somalia","2009-01-01","2009-07-31","9-59M",1,"NaN"),
("somalia","2009-08-01","2009-09-01","9-59M",1,"Sub-national, Somaliland (Awdal, Galbeed, Sahe..."),
("somalia","2010-01-01","2010-06-01","9-59M",1,"Somaliland & Puntland, South and Central Zone"),
("somalia","2011-07-01","NaT","9-59M",1,"Somaliland - CHD 1st Round"),
("somalia","2011-07-01","NaT","9-59M",1,"Puntland - CHD 1st Round"),
("somalia","2011-04-01","2011-04-28","9-59M",1,"Garow"),
("somalia","2011-07-01","2011-07-31","6M-15Y",1,"Mogadishu"),
("somalia","2011-08-01","2011-08-31","6M-15Y",1,"Central Zone Mogadishu"),
("somalia","2011-09-01","2011-09-01","6M-15Y",1,"South Zone Gedo Mop up emergency"),
("somalia","2011-10-01","2011-11-01","6M-14Y",1,"Mopup Emergency  Mogadishu 14 out of 16 districts"),
("somalia","2011-08-01","2011-11-01","6M-14Y",1,"Emergency Response Vaccination"),
("somalia","2012-02-01","2012-05-01","9-59M",1,"Round 1"),
("somalia","2012-02-01","2012-03-31"," 6-59M",1,"NaN"),
("somalia","2012-10-01","2012-11-30","9-59M",1,"Round 2"),
("somalia","2013-12-22","2013-12-27","9-59M",1,"with OPV"),
("somalia","2014-10-11","2014-11-01","9-59M",1," Mogadishu 11 - 15 Oct\r\nPuntland 28 Oct - 1 Nov"),
("somalia","2015-11-15","2016-01-04"," 9M-9Y",1,"NaN"),
("somalia","2016-08-20","2016-08-23","9-59M",1,"NaN"),
("somalia","2016-12-16","NaT","9-59M",1,"Integrated with VitA"),
("somalia","2017-01-01","2017-03-31"," 6-59M",1,"NaN"),
("somalia","2017-05-01","NaT"," 6M-5Y",1,"NaN"),
("somalia","2018-01-02","2018-01-06","6M-10Y",1,"and from 11 to 22 March"),
("somalia","2019-11-24","2019-11-28"," 6-59M",1,"or 6M-10Y: TP 5,980,401"),
]

## Madagascar
sia_calendar += [
("madagascar","2004-09-13","2004-10-08","M-14Y",1,"NaN"),
("madagascar","2007-10-22","2007-10-30","9-59M",1,"NaN"),
("madagascar","2010-10-25","2010-10-29","9-47M",1,"NaN"),
("madagascar","2013-10-14","2013-10-24","9-59M",1,"NaN"),
("madagascar","2016-10-17","2016-10-21","9-59M",1,"75% of districts attained 95% admin coverage\n..."),
("madagascar","2018-11-01","NaT","9M-5Y",1,"NaN"),
("madagascar","2019-01-14","2019-01-18","9M-9Y",1,"Phase 1: 25 districts - Phase 2: 22 districts,..."),
("madagascar","2019-02-18","2019-02-22","6M-9Y",1,"22 districts"),
("madagascar","2019-03-25","2019-04-05","6M-9Y",1,"67 districts"),
("madagascar","2022-05-16","2022-05-22","9-59M",1,"NaN"),
]

## Format the calendar as a dataframe
sia_calendar = pd.DataFrame(sia_calendar,
                            columns=["country","start_date","end_date",
                                     "age_group","target_pop","notes"])
sia_calendar["start_date"] = pd.to_datetime(sia_calendar["start_date"])
sia_calendar["end_date"] = pd.to_datetime(sia_calendar["end_date"])

if __name__ == "__main__":

    ## Get the list of countries
    countries = sorted(list(countries))

    ## Create multiindex series with the appropriate shape
    time_index = pd.date_range(start="2008-12-31",
                               end="2023-12-31",
                               freq="SM")
    index = pd.MultiIndex.from_product([countries,time_index],
                                        names=["country","time"])
    target_pop = pd.Series(np.zeros((len(index),)),
                           index=index,
                           name="target_pop")

    ## Loop over SIAs and fill in the data at the
    ## desired admin level
    for i, sia in sia_calendar.iterrows():

        ## Start by getting the closest time to the intended
        ## time index. We use the start date since a lot of the
        ## entries in the dataset have no end date.
        sia_time = time_index[np.argmin(np.abs(time_index-sia.loc["start_date"]))]

        ## Get the appropriate rows
        rows = [(c, sia_time) for c in countries if c.startswith(sia.loc["country"])]

        ## Set the values
        target_pop.loc[rows] = sia.loc["target_pop"]

    ## Serialize the result
    target_pop.to_pickle("..\\..\\outputs\\target_pop.pkl")
    print(target_pop.loc[target_pop != 0])