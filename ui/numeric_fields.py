"""Keep FilmClear's parameter inputs readable in Chinese Hong Kong and Macau locales."""

from PySide6.QtCore import QLocale


def use_standard_digits(spinbox):
    locale = spinbox.locale()
    # Windows can report 〇 as its native zero, which Qt pairs with 〡–〩.
    chinese_hk_or_mo = (locale.language() == QLocale.Chinese
                        and locale.territory() in (QLocale.HongKong, QLocale.Macao))
    if locale.zeroDigit() == "〇" or chinese_hk_or_mo:
        spinbox.setLocale(QLocale.c())
