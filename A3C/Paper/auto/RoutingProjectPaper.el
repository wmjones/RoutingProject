(TeX-add-style-hook
 "RoutingProjectPaper"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "margin=1in") ("apacite" "natbibapa")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "geometry"
    "titlesec"
    "amsmath"
    "algorithm"
    "algpseudocode"
    "graphicx"
    "caption"
    "accents"
    "amssymb"
    "flexisym"
    "siunitx"
    "mathtools"
    "hyperref"
    "apacite")
   (TeX-add-symbols
    '("norm" 1)
    '("ubar" 1))
   (LaTeX-add-labels
    "intro"
    "fig_sr_trainrate"
    "fig_samp_var"
    "fig_sl_table")
   (LaTeX-add-bibliographies
    "bibliography.bib")
   (LaTeX-add-mathtools-DeclarePairedDelimiters
    '("floor" "")))
 :latex)

