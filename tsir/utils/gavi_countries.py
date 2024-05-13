""" gavi_countries.py

Simple data structures for the gavi countries getting measles support. See
https://www.gavi.org/programmes-impact/our-impact/countries-approved-support
for more information.

Note: Names are harmonized with the WHO spreadsheet by hand. """

## Countries with MCV SIAs
msia = {"afghanistan","chad","democratic republic of the congo",
		"ethiopia","nigeria","pakistan"}

## Countries with MR campaigns
mr = {"bangladesh","burkina faso","burundi","cambodia","cameroon",
	  "gambia","ghana","india","indonesia","kenya","lesotho","malawi",
	  "myanmar","papua new guinea","rwanda","sao tome and principe",
	  "senegal","solomon islands","united republic of tanzania","viet nam",
	  "yemen","zambia","zimbabwe"}

## Countries with second dose campaigns
msd = {"bangladesh","burkina faso","burundi","cambodia","eritrea","gambia",
	   "ghana","democratic people's republic of korea","malawi","mozambique","myanmar",
	   "nepal","papua new guinea","rwanda","sao tome and principe","senegal",
	   "sierra leone","united republic of tanzania","viet nam","zambia","zimbabwe"}

## Extras
extras = {"somalia","madagascar"}

## All together
countries = msia.union(mr).union(msd).union(extras)