import numpy as np
if __name__ == '__main__':
    head='''
\\documentclass[a4paper,12pt]{elsarticle}
\\usepackage[top=2.5cm, bottom=2cm, left=2cm, right=2cm]{geometry}
\\usepackage{amsmath}
\\usepackage{fullpage}
\\usepackage{amssymb}
\\usepackage[usenames]{color}
\\usepackage{latexsym}
\\usepackage{subfigure}
\\usepackage[mathscr]{eucal}
\\pagestyle{empty}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.16}
\\usetikzlibrary{math,fit,positioning}
\\graphicspath{{../figure/}}
\\setlength{\\fboxrule}{1pt}
\\setlength{\\fboxsep}{0.000001pt}
\\begin{document}

\\begin{figure}[!htbp]
    \\centering
    \\begin{tikzpicture}
    %\\small
    % 定义自定义颜色
    \\definecolor{CustomBoarder}{RGB}{33,33,33} % Morandi Brown
    \\definecolor{CustomFill}{RGB}{255,255,255}  % Oliver green
    \\definecolor{CustomFill1}{RGB}{142,167,136} 
    \\definecolor{CustomFill2}{RGB}{136,162,132} %  Red
    \\definecolor{CustomFill3}{RGB}{142,167,136}  % Oliver green
    \\definecolor{CustomFill4}{RGB}{132,182,191} % sea blue
    \\definecolor{CustomFill5}{RGB}{224,185,184} % Pink
    \\begin{axis}[
      xlabel={Complex\hskip 60pt $1/H(s)$ \hskip 60pt Simple},
      ylabel={Low \hskip 30pt Degree of SSH \hskip 30pt High},
      xmin=-0.3, xmax=1.2,
      ymin=-0.02, ymax=0.15,
      yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3}, % 设置y轴数值格式
      scaled y ticks=false, % 关闭y轴的缩放（即科学记数）
      axis lines=middle,
      width=9.6cm,
      height=8cm,
      grid=both,
      grid style={line width=.1pt, draw=gray!30},
      major grid style={line width=.2pt,draw=gray!30},
      minor grid style={line width=.1pt,draw=gray!10},
      minor tick num=4,
      enlargelimits={abs=0.001},
      samples=2,
      domain=0:3.0,
      legend style={at={(0.5,-0.02)},anchor=north,legend columns=-1},
      every axis plot post/.append style={ultra thick},
      clip=false,
      xlabel style={at={(current axis.south)}, anchor=north, below=0pt},
      ylabel style={at={(current axis.west)}, rotate=90, anchor=south, yshift=5pt}
      ]
      \\addplot[only marks, mark=*, mark size=2pt, color=black] coordinates
    % Points
    {
    '''
    #读取点数据
    points=np.loadtxt("/tmp/points.csv")
    # Add points:
    for i in range(points.shape[0]-1):
        p=points[i,:]
        head+="("+str(1.0/p[1])+","+str(p[0])+") "
    p=points[points.shape[0]-1,:]
    head+="("+str(1.0/p[1])+","+str(p[0])+")};\n"
    for i in range(points.shape[0]):
        p=points[i,:]
        head+="\\fill[CustomFill3, opacity=0.1] (axis cs:0,0) rectangle (axis cs: "+str(1.0/p[1])+","+str(p[0])+");\n"
        head+="\\draw[CustomBoarder] (axis cs:0,0) rectangle (axis cs:" +str(1.0/p[1])+","+str(p[0])+");\n"
        if i<3 or i==points.shape[0]-1:
            head+="\\node at (axis cs:"+str(1.0/p[1])+","+str(p[0])+") [above] {"+str(i+1)+"};\n"
    head+='''
    \\end{axis}
    \\end{tikzpicture}
    \\end{figure}
    \\end{document}
    '''
    with open('/tmp/a.tex', 'w') as f:
        print(head, file=f,flush=True)
    import os
    curdir=os.getcwd()
    os.system("cd /tmp; rm *.pdf *.eps *.ps *.tif; pdflatex a.tex; pdfcrop a.pdf; mv a-crop.pdf a.pdf; mv a.pdf \""+ curdir +"\"  > /dev/null 2>&1")
