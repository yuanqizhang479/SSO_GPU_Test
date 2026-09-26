#!/usr/bin/env python3
"""Generate/check high-precision numerical JSON and manuscript LaTeX snapshots.

80 working digits; the non-diagonal experiment is repeated at 100 digits.
This is numerical verification, not interval arithmetic or a formal proof.
"""
if not __debug__:
    raise RuntimeError('Verification requires assertions: rerun without -O / PYTHONOPTIMIZE.')
import argparse
import json
from pathlib import Path
import mpmath as mp
ROOT=Path(__file__).resolve().parent
mp.mp.dps=80

def require(ok,msg):
    if not ok: raise ArithmeticError(msg)

def root(f,lo,hi):
    lo,hi=mp.mpf(lo),mp.mpf(hi)
    require(f(lo)<0<f(hi),'Root is not bracketed')
    for _ in range(int(mp.mp.dps*3.5)+12):
        mid=(lo+hi)/2
        if f(mid)<0:lo=mid
        else:hi=mid
    return (lo+hi)/2

def inner(a,b):return mp.fsum(a[i,j]*b[i,j] for i in range(a.rows) for j in range(a.cols))
def frob(a):return mp.sqrt(inner(a,a))
def z2(a,d):
    det=mp.det(a);n=inner(a,a);q=mp.sqrt(det**2+d*d*n+d**4)
    at=mp.matrix([[a[1,1],-a[1,0]],[-a[0,1],a[0,0]]])
    return (det*at+(d*d+q)*a)/(q*mp.sqrt(n+2*d*d+2*q))
def zg(a,d):
    vals,q=mp.eigsy(a.T*a+d*d*mp.eye(a.cols))
    return a*q*mp.diag([1/mp.sqrt(v) for v in vals])*q.T

def non_diag(dps):
    with mp.workdps(dps):
        R=lambda n,d=1:mp.mpf(n)/d
        blocks=[(R(1),R(1,6),R(2,5),R(1,5),R(2,3)),(R(3,5),R(1,6),R(3,10),R(1,10),R(1,3))]
        theta=mp.matrix([[R(1,6),0,R(2,5),0],[0,R(1,6),0,R(3,10)],[R(1,5),0,R(2,3),0],[0,R(1,10),0,R(1,3)]])
        def field(t,d):
            out=mp.matrix(4)
            for ids,(r,a,b,c,e) in zip(((0,2),(1,3)),blocks):
                A=mp.matrix([[r+a*t*d,b*t*d],[c*t*d,e*t*d]])
                # Analytic q/d continuation of the exact 2x2 formula at d=0.
                det_over_d=r*e*t+(a*e-b*c)*t*t*d
                L=mp.sqrt(det_over_d**2+inner(A,A)+d*d)
                at=mp.matrix([[A[1,1],-A[1,0]],[-A[0,1],A[0,0]]])
                Z=(det_over_d*at+(d+L)*A)/(L*mp.sqrt(inner(A,A)+2*d*d+2*d*L))
                for i in range(2):
                    for j in range(2):out[ids[i],ids[j]]=Z[i,j]
            return out
        h=lambda t,d:inner(theta,field(t,d))
        t0=root(lambda t:h(t,R(0)),-2,0)
        t1=-mp.diff(lambda d:h(t0,d),R(0))/mp.diff(lambda t:h(t,R(0)),t0)
        MR=mp.matrix(4)
        for i in range(4):
            for j in range(4):MR[i,j]=mp.diff(lambda d:field(t0,d)[i,j],R(0))+t1*mp.diff(lambda t:field(t,R(0))[i,j],t0)
        phi=field(t0,R(0));rows=[]
        for dt in ('1e-2','1e-3','1e-4','1e-5','1e-6'):
            d=mp.mpf(dt);t=root(lambda t:h(t,d),-2,0);Z=field(t,d);scaled=(Z-phi)/d
            A=mp.diag([1,R(3,5),0,0])+t*d*theta
            require(frob(Z-zg(A,d))<mp.mpf(10)**(-dps+20),'Block and general Gram formulas disagree')
            rows.append(dict(delta=d,multiplier_ratio=t,direction_ratio=frob(scaled),finite_t1=(t-t0)/d,cosine=inner(scaled,MR)/(frob(scaled)*frob(MR)),tangency=abs(h(t,d))))
        return dict(t0=t0,t1=t1,mr_norm=frob(MR),mr=[[MR[i,j] for j in range(4)] for i in range(4)],rows=rows)

def fixed_data():
    rows=[];R=lambda n,d=1:mp.mpf(n)/d
    cases=[('regular',R(1),R(1,2),R(1,2),R(2),('1e-4','1e-5'),-R(16,9)),('strict',R(1,2),R(1),R(1,4),R(1),('1e-6','1e-8'),-R(5)/(4*mp.sqrt(15))),('boundary',R(1),R(1),R(1),R(2,3),('1e-9','1e-12'),mp.root(2,3))]
    for name,b,c,ls,power,ds,limit in cases:
        for dt in ds:
            d=mp.mpf(dt);lam=root(lambda l:z2(mp.matrix([[l,b],[b,c]]),d)[0,0],-2,2)
            ratio=(lam-ls)/d**power
            if name=='boundary':ratio=-ratio
            rows.append(dict(regime=name,delta=d,ratio=ratio,limit=limit))
    return rows

def profile_data():
    def h(x,d,profile,L):
        lam=1-x;disc=mp.sqrt((lam-1)**2+4);lp,lm=(lam+1+disc)/2,(lam+1-disc)/2
        if profile.startswith('huber'):fn=lambda v:max(-mp.mpf(1),min(mp.mpf(1),v/(L*d)))
        else:fn=lambda v:v/(d**3+abs(v)**3)**(mp.mpf(1)/3)
        return fn(lp)*(lam-lm)/disc+fn(lm)*(lp-lam)/disc
    rows=[]
    for name,ds,power,limit,L in [('huber',('1e-4','1e-6','1e-8'),mp.mpf(1),mp.mpf(2),mp.mpf(1)),('p3',('1e-3','1e-6','1e-9','1e-12'),mp.mpf(3)/4,mp.root(mp.mpf(8)/3,4),mp.mpf(1)),('huber_L_half',('1e-8',),mp.mpf(1),mp.mpf(1),mp.mpf('0.5'))]:
        for dt in ds:
            d=mp.mpf(dt);x=root(lambda x:-h(x,d,name,L),0,1)
            rows.append(dict(profile=name,delta=d,ratio=x/d**power,limit=limit))
    return rows

def corank_data():
    R=lambda n,d=1:mp.mpf(n)/d
    eta=root(lambda e:e/mp.sqrt(1+e*e)+(e/4)/mp.sqrt(1+e*e/4)-R(3,4),0,10)
    kr=mp.matrix([1,1,-eta/mp.sqrt(1+eta*eta),-(eta/2)/mp.sqrt(1+eta*eta/4)])
    kf=mp.matrix([1,1,-R(3,5),-R(3,10)]);rows=[]
    for dt in ('1e-2','1e-3','1e-4','1e-5'):
        d=mp.mpf(dt)
        def phi(t):return mp.matrix([v/mp.sqrt(v*v+d*d) for v in (1+R(3,8)*t*d,1+R(3,8)*t*d,t*d,t*d/2)])
        coeff=mp.matrix([R(3,8),R(3,8),1,R(1,2)]);t=root(lambda t:inner(coeff,phi(t)),-2,0);z=phi(t)
        rows.append(dict(delta=d,displacement=t*d,error_R=frob(z-kr),error_F=frob(z-kf)))
    return dict(eta=eta,kr=[kr[2],kr[3]],center_distance=frob(kr-kf),rows=rows)

def compact_data():
    rows=[]
    for dt in ('1e-2','1e-3','1e-4','1e-5'):
        d=mp.mpf(dt)
        def phi(t):
            s=t*d**3;return mp.matrix([(2+s)/mp.sqrt((2+s)**2+d*d),(1-s)/mp.sqrt((1-s)**2+d*d),s/mp.sqrt(s*s+d*d)])
        t=root(lambda t:phi(t)[0]-phi(t)[1]+phi(t)[2],-1,0)
        rows.append(dict(delta=d,multiplier_ratio=t,direction_ratio=frob(phi(t)-mp.matrix([1,1,0]))/d**2))
    return dict(limit_multiplier=-mp.mpf(3)/8,limit_direction=mp.sqrt(26)/8,rows=rows)

def quintic_data():
    R=lambda n,d=1:mp.mpf(n)/d
    th=mp.matrix([[R(27,10),0,0,R(3,10)],[0,-R(16,5),0,R(1,10)],[0,0,R(1,2),R(1,4)],[R(1,5),-R(3,20),R(1,10),1]])
    A=mp.diag([3,2,1,0]);target=mp.diag([1,1,1,0]);J=mp.diag([R(1,9),R(1,4),1,0]);rows=[]
    for dt in ('1e-2','1e-3','1e-4','1e-5'):
        d=mp.mpf(dt);t=root(lambda t:inner(th,zg(A+t*d**5*th,d)),-1,0);Q=(zg(A+t*d**5*th,d)-target+d*d*J/2)/d**4
        rows.append(dict(delta=d,multiplier_ratio=t,quartic_diagonal=[Q[i,i] for i in range(4)],quartic_offdiag=mp.sqrt(mp.fsum(Q[i,j]**2 for i in range(4) for j in range(4) if i!=j))))
    return rows

def crossover_data():
    import verify_unified_crossover as v
    D=v.D;rows=[]
    for z in (0,1,3,10):
        d=D('1e-9');gap=D(z)*D('1e-6');x,xh,b,k,derr=v.solve_path(gap,d)
        rows.append(dict(path='critical',parameter=str(z),delta=str(d),gap=str(gap),normalized=str(x/D('1e-6')),predicted=str(v.predicted_y(D(z))),unified_ratio=str(x/xh),direction_ratio=str(derr/x)))
    for k in range(2,6):
        q=D(10)**(-k);d=q**3;x,xh,b,kap,derr=v.solve_path(q,d)
        rows.append(dict(path='strict',parameter=str(k),delta=str(d),gap=str(q),normalized=str(x*q.sqrt()/d),predicted='1',unified_ratio=str(x/xh),direction_ratio=str(derr/x)))
    for k in range(2,11):
        q=D(10)**(-k);d=q**3;mode=k%3;gap=D(0) if mode==0 else (D(3)*q*q if mode==1 else q)
        x,xh,b,kap,derr=v.solve_path(gap,d)
        rows.append(dict(path='oscillatory',parameter=str(k),window=('boundary' if mode==0 else 'critical' if mode==1 else 'strict'),delta=str(d),gap=str(gap),normalized=str(x/xh),predicted='1',unified_ratio=str(x/xh),direction_ratio=str(derr/x)))
    d=D('1e-15');x,xh,b,k,derr=v.solve_path(D(0),d)
    rows.append(dict(path='boundary',parameter='0',delta=str(d),gap='0',normalized=str(x/D('1e-10')),predicted=str(D(2)**(D(1)/D(3))),unified_ratio=str(x/xh),direction_ratio=str(derr/x)))
    return rows

def serial(x):
    if isinstance(x,mp.mpf):return mp.nstr(x,60)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,list):return [serial(v) for v in x]
    return x

def fmt(x,n=10):
    x=mp.mpf(x);number=str(int(mp.floor(abs(x)*mp.mpf(10)**n+mp.mpf('.5')))).zfill(n+1)
    return ('-' if x<0 else '')+number[:-n]+'.'+number[-n:]
def sci(x,n=6):
    x=mp.mpf(x)
    if not x:return '0'
    e=int(mp.floor(mp.log10(abs(x))));return fmt(x/mp.mpf(10)**e,n)+r'\times10^{'+str(e)+'}'
def dtex(x):return r'10^{'+str(int(mp.nint(mp.log10(mp.mpf(x)))))+'}'
def math(x):return r'\('+x+r'\)'
def table(headers,rows):
    return '\\begin{center}\n\\begin{tabular}{@{}'+'l'*len(headers)+'@{}}\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+r'\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{center}\n'

def latex(data):
    macros={}
    def add(k,x,n=10):macros[k]=fmt(x,n)
    last={r['regime']:r for r in data['fixed']}
    for label,key in [('Regular','regular'),('Strict','strict'),('Boundary','boundary')]:add(label+'Ratio',last[key]['ratio']);add(label+'Limit',last[key]['limit'])
    for r in data['profiles']:
        if r['profile']=='huber' and r['delta']==mp.mpf('1e-8'):add('HuberBoundaryRatio',r['ratio'])
        if r['profile']=='p3' and r['delta']==mp.mpf('1e-9'):add('PowerThreeRatio',r['ratio']);add('PowerThreeLimit',r['limit'])
    c=data['corank'];add('CorankSlopeFour',c['rows'][2]['displacement']/c['rows'][2]['delta'],7);add('CorankEta',c['eta'],12);add('CenterDistance',c['center_distance'],12);add('CorankKrOne',-c['kr'][0],12);add('CorankKrTwo',-c['kr'][1],12)
    n=data['nondiagonal'];add('NonDiagonalEta',-n['t0'],12);add('NonDiagonalTOne',n['t1']);add('NonDiagonalMR',n['mr_norm'])
    for r in n['rows']:
        suffix='Five' if r['delta']==mp.mpf('1e-5') else 'Six' if r['delta']==mp.mpf('1e-6') else None
        if suffix:
            for name,key in [('Direction','direction_ratio'),('Slope','multiplier_ratio'),('FiniteTOne','finite_t1'),('Cosine','cosine')]:add('NonDiagonal'+name+suffix,r[key],12)
    co=data['compact'];r=co['rows'][2];add('CompactMultiplier',r['multiplier_ratio']);add('CompactDirection',r['direction_ratio']);add('CompactDirectionLimit',co['limit_direction'])
    add('BoundaryDirection',data['crossover'][-1]['direction_ratio'],12);add('BoundaryDirectionLimit',mp.sqrt(mp.mpf(3)/2),12)
    q=data['quintic'][2]
    for i,x in enumerate(q['quartic_diagonal']):add('Quartic'+('One','Two','Three','Four')[i],x,12)
    out={'numerical_macros.tex':'% Generated by verify_publication_tables.py; do not hand-edit.\n'+'\n'.join(r'\newcommand{\num'+k+'}{'+v+'}' for k,v in sorted(macros.items()))+'\n'}
    out['table_fixed.tex']=table(['Regime',math(r'\delta'),'Normalized displacement','Limit'],[[r['regime'],math(dtex(r['delta'])),math(fmt(r['ratio'])),math(fmt(r['limit']))] for r in data['fixed']])
    out['table_profiles.tex']=table(['Profile',math(r'\delta'),'Normalized displacement','Limit'],[[{'p3':math('p=3'),'huber':'Huber','huber_L_half':math('L=1/2')}[r['profile']],math(dtex(r['delta'])),math(fmt(r['ratio'],12)),math(fmt(r['limit'],12))] for r in data['profiles']])
    out['table_corank.tex']=table([math(r'\delta'),math(r'\lambda_\delta-\lambda^\star'),math(r'\|\Phi_\delta-\Phi_R^\star\|_F'),math(r'\|\Phi_\delta-\Phi_F^\star\|_F')],[[math(dtex(r['delta']))]+[math(sci(r[k])) for k in ('displacement','error_R','error_F')] for r in c['rows']])
    out['table_nondiagonal.tex']=table([math(r'\delta'),math(r't_\delta'),math(r'\|\Phi_\delta-\Phi_R^\star\|_F/\delta'),math(r'(t_\delta-t_0)/\delta')],[[math(dtex(r['delta']))]+[math(fmt(r[k],12)) for k in ('multiplier_ratio','direction_ratio','finite_t1')] for r in n['rows']])
    out['table_quintic.tex']=table([math(r'\delta'),math(r'(\lambda_\delta-1)/\delta^5')],[[math(dtex(r['delta'])),math(fmt(r['multiplier_ratio']))] for r in data['quintic']])
    for mode in ('critical','strict','oscillatory'):
        rs=[r for r in data['crossover'] if r['path']==mode]
        if mode=='critical':headers=[math(r'\zeta'),math(r'\delta'),math(r'x_\delta/\delta^{2/3}'),math(r'y_\zeta')];keys=['parameter','delta','normalized','predicted']
        elif mode=='strict':headers=[math(r'\delta'),math(r'x_\delta\sqrt{\Delta_\delta}/\delta'),math(r'x_\delta/\widehat x_\delta')];keys=['delta','normalized','unified_ratio']
        else:headers=[math('k'),'Window',math(r'\delta_k'),math(r'x_{\delta_k}/\widehat x_{\delta_k}')];keys=['parameter','window','delta','unified_ratio']
        rows=[[r[k] if k in ('parameter','window') else math(dtex(r[k]) if k=='delta' else fmt(r[k],12)) for k in keys] for r in rs]
        out['table_'+mode+'.tex']=table(headers,rows)
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    print('Computing 80-digit tables; repeating non-diagonal example at 100 digits.',flush=True)
    n80,n100=non_diag(80),non_diag(100)
    for key in ('t0','t1','mr_norm'):require(abs(n80[key]-n100[key])<mp.mpf('1e-65'),'Precision disagreement: '+key)
    for a,b in zip(n80['rows'],n100['rows']):
        for key in ('multiplier_ratio','direction_ratio','finite_t1','cosine'):require(abs(a[key]-b[key])<mp.mpf('1e-60'),'Precision disagreement: '+key)
    data=dict(fixed=fixed_data(),profiles=profile_data(),corank=corank_data(),nondiagonal=n80,compact=compact_data(),quintic=quintic_data(),crossover=crossover_data())
    r=n80['rows'][3];require(abs(r['finite_t1']-mp.mpf('0.70104739544355205579'))<mp.mpf('1e-20'),'Finite t1 regression')
    require(abs(r['direction_ratio']-mp.mpf('0.671768177074'))<mp.mpf('5e-13'),'Direction regression 1e-5')
    require(abs(n80['rows'][4]['direction_ratio']-mp.mpf('0.671774660292'))<mp.mpf('5e-13'),'Direction regression 1e-6')
    require(abs(n80['t1']-mp.mpf('0.7010530600'))<mp.mpf('5e-10'),'Intrinsic t1 regression')
    ps=[r for r in data['profiles'] if r['profile']=='p3'];require(all(abs(ps[i+1]['ratio']-ps[i+1]['limit'])<abs(ps[i]['ratio']-ps[i]['limit']) for i in range(len(ps)-1)),'p3 convergence regression')
    metadata=dict(working_digits=80,nondiagonal_repeat_digits=100,nondiagonal_agreement_absolute='1e-60 ratios; 1e-65 constants',precision_warning='Cross-precision agreement is not a validated error bound.',dependencies={'mpmath':mp.__version__},data=serial(data))
    files={ROOT/'results'/'publication_numbers.json':json.dumps(metadata,indent=2)+'\n'}
    files.update({ROOT/'submission'/'generated'/k:v for k,v in latex(data).items()})
    for path,text in files.items():
        if args.check:require(path.exists() and path.read_text()==text,'Snapshot differs: '+str(path.relative_to(ROOT)))
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    print(('Checked' if args.check else 'Wrote')+f' {len(files)} publication snapshots. High-precision regressions passed.')
if __name__=='__main__':main()
