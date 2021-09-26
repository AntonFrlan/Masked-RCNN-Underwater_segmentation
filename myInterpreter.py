def uBroj(naziv):
    if naziv == "Optezivac":
        return 1
    elif naziv == "Cijev":
        return 2
    elif naziv == "More":
        return 3
    else:
        raise ValueError("Nije uspio pretvoriti BROJ u NAZIV!!!!")


def uNaziv(broj):
    if broj == 1:
        return "Optezivac"
    elif broj == 2:
        return "Cijev"
    elif broj == 3:
        return "More"
    else:
        raise ValueError("Nije uspio pretvoriti NAZIV u BROJ!!!!")