\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{misqdoc}[2014/03/01 MISQ Manuscript Stylefile]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[12pt,letterpaper]{article}

% Required packages and font settings
\RequirePackage{xltxtra}
\RequirePackage{fontspec}
\RequirePackage{xunicode}
\RequirePackage{graphicx}
\RequirePackage[labelsep=period,font=bf]{caption}
\RequirePackage[margin=1in]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage[doublespacing]{setspace}
\setstretch{2.1}
\RequirePackage{enumitem}
\setlist{noitemsep}
\RequirePackage[document]{ragged2e}
\defaultfontfeatures{Scale=MatchLowercase}
\setmainfont[Ligatures={Common,TeX}]{Times New Roman}
\RequirePackage[backend=biber, style=apa]{biblatex}

% Author and affiliation formatting using authblk package
\RequirePackage{authblk}
\renewcommand*{\Authsep}{, }
\renewcommand*{\Authand}{, }
\renewcommand*{\Authands}{, }
\renewcommand*{\Affilfont}{\normalsize}
\setlength{\affilsep}{2em}

% Font sizes for various sections
\renewcommand{\normalsize}{\fontsize{12pt}{12pt}\selectfont}
\renewcommand{\Huge}{\fontsize{20pt}{24pt}\selectfont}
\renewcommand{\LARGE}{\fontsize{13pt}{15pt}\selectfont}
\renewcommand{\Large}{\fontsize{12pt}{14pt}\selectfont}
\renewcommand{\large}{\fontsize{11pt}{13pt}\selectfont}
\setlength{\parindent}{15pt}
\setlength{\parskip}{0pt}

% Short title for headers
\let\@shorttitle\@empty
\newcommand{\shorttitle}[1]{\gdef\@shorttitle{#1}}

% Custom maketitle command
\renewcommand{\maketitle}{%
  \begin{center} %
    {\Huge\bfseries\@title\par}
    \vspace{10pt} % Add some space after the title
    \Authfont % Apply author font settings
    \begin{tabular}[t]{c} % Alignment and tabular for author names
      \Authand % Author separator
    \end{tabular}\\[10pt] % Space between author names and affiliations
    \Affilfont % Apply affiliation font settings
    \begin{tabular}[t]{c} % Alignment and tabular for affiliations
      \@affilsep % Affiliation separator
    \end{tabular}
  \end{center}
  \thispagestyle{plain}\@thanks
}

% Section formatting
\renewcommand{\section}{%
  \@startsection
    {section}{1}{0pt}{4pt}%
    {4pt}{\bfseries\LARGE\centering\MakeUppercase}%
}
\renewcommand{\subsection}{%
  \@startsection
    {subsection}{2}{0pt}{4pt}%
    {4pt}{\bfseries\large\centering}%
}
\renewcommand{\subsubsection}{%
  \@startsection
    {subsubsection}{3}{0pt}{4pt}%
    {4pt}{\bfseries}%
}
\setcounter{secnumdepth}{1}

% Header and footer settings
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textit{\@shorttitle}}
\fancyfoot[R]{\textit{In Preparation for MISQ}\hspace{5mm}\textbf{\thepage}}
\fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[R]{\textit{In Preparation for MISQ}\hspace{5mm}\textbf{\thepage}}
}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
