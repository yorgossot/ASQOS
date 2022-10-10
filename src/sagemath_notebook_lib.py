from . import notebook_library
import sage.all as sg
mathematica_to_sagemath = {'Im':sg.imag_part,"Re":sg.real,"E":sg.exp, "Sqrt":sg.sqrt }
def separate_simpl_compl_factors(expr):
    expr_factors = []
    len_factor = []

    factors = expr.operands()
    for i,fac in enumerate(factors):
        expr_factors.append(str(fac))
        len_factor.append(len(str(fac)))

    max1 = sorted(len_factor)[::-1][0]
    max2 = sorted(len_factor)[::-1][1]

    ind_max1 = len_factor.index(max1)
    ind_max2 = len_factor.index(max2)


    simple_factor =1
    compl_factor = 1
    for i,fac in enumerate(factors):
        if i in [ind_max1,ind_max2]:
            compl_factor = compl_factor*fac
        else:
            simple_factor = fac*simple_factor
    return simple_factor , compl_factor
def MMA_collect(expr,coll):
    return expr._mathematica_().Collect(coll)._sage_(locals=mathematica_to_sagemath).subs(E=sg.e)

        
        #Discard code below?
        
# def p_subs(expr):
    
#     #P1
#     fac = [1,2]
#     for f in fac:    
#         expr = expr.subs(f*(12*c)== f*(P1 -16*c^2 -1) )
#     #P2
#     fac = [1,2*sg.I,sg.I]
#     for f in fac:   
#         expr = expr.subs(f*(48*c^2)== f*(P2 -16*c -1))

#     #c3
#     fac = [1,32*I,-32*I,64*I,-16*I]
#     for f in fac:  
#         expr = expr.subs( f*(c+1/8)==f*c3)
#     #c2
#     fac = [1,-16*I,-8*I]
#     for f in fac:  
#         expr = expr.subs( f*(c+1/4)==f*I*c2)
    
#     return expr

# '''
#     sg.var('Cts',latex_name=r"\tilde{C}_s")
#     sg.var('Ctp',latex_name=r"\tilde{C}_p")
#     ct0_val = 1/2*( Cts+ sg.sqrt(Cts^2+4*Ctp) )
#     ct1_val = 1/2*( Cts- sg.sqrt(Cts^2+4*Ctp) )
#     sg.var('c2','c3')

#     sg.var('P1','P2','P3')



#     polynomials = [[16,12,1],
#                 [48,16,1],]
# '''


# def custom_simplify(expr,ct0_val,ct1_val):
#     '''
#     Only for elements that have both  0 and 1 dependency


#     '''
#     expr = notebook_library.symround(expr)
    
#     expr = expr.subs(Ct0=ct0_val,Ct1=ct1_val).expand()

    
#     expr = notebook_library.symround(notebook_library.MMA_simplify(expr))
#     expr = MMA_collect(expr,'{CtE,Ctp,Cts}')
#     expr = p_subs(expr)
     
    
#     return expr
    