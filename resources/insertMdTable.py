def insertMdTable(dfin, name, mdtext, plot=False, transpose=False, maxcols=10):
   
    import numpy as np
    from IPython.display import Markdown as md
    from tabulate import tabulate
    
    nc = int(np.ceil(dfin.shape[int(not transpose)]/maxcols))
 
    dfin = dfin.apply(lambda x: x.astype(str) if not np.issubdtype(x, np.number) 
                                else x.astype(int) if x.abs().max()>100
                                else x.round(3) if x.abs().max()>0.1
                                else x.round(6) if x.abs().max()>0.0001
                                else x.round(9) if x.abs().max()>0.000001 
                                else x, axis=0)
    
    tmp_mdtxt = ""
    for i in range(nc):
        if transpose:  
            df = dfin.iloc[(i*maxcols):(min([(i+1)*maxcols,dfin.shape[1]])),:]
            if i<nc-1:
                df['...'] = ''
            table = df.T
            table.index.name = ' '
            table = table.reset_index()
            try:
                table = table.to_markdown(floatfmt="0.12g", index=False)
            except:
                table = tabulate(table, table.columns, floatfmt="0.12g", tablefmt='pipe', index=False)
        else:
            df = dfin.iloc[:,(i*maxcols):(min([(i+1)*maxcols,dfin.shape[1]]))]
            if i<nc-1:
                df['...'] = ''
            try:
                table = df.to_markdown(floatfmt="0.12g", index=False)
            except:
                table = tabulate(df, df.columns, floatfmt="0.12g", tablefmt='pipe', index=False)
            
            table = table+'\n'
        
        if i!=nc-1:
            tmp_mdtxt = tmp_mdtxt + table.replace("|\n","\n") + "\n\n"
        else:
            tmp_mdtxt = tmp_mdtxt + table

    mdtext = mdtext.replace(name+'.table', tmp_mdtxt)
    
    if plot:
        print(tmp_mdtxt)
    return mdtext