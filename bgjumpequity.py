'''
Copyright 2012 Mark Higgins

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


This script contains functionality for calculating cubeful equity
when assuming a jump model for win probability evolution.

Self-consistently solves for cube decision points and
properly handles initial doubles, redoubles, and too-good-to-double
points.

It requires that the scipy package be installed. scipy can be
downloaded from here: http://www.scipy.org/Download
'''

import bisect
import math
import scipy
from scipy.stats import norm

def _root( Es, Ps, a ):
    '''Finds the root where E=a, assuming piecewise linear.'''

    index = bisect.bisect_left( Es, a )
    if index == 0:
        return 0
    elif index == len(Es):
        return 1
    else:
        return ( Ps[index-1]*(Es[index]-a) - Ps[index]*(Es[index-1]-a) ) / ( Es[index]-Es[index-1] )

def _interpParam( P, jumpCutoff, midParam ):
    '''Interpolates a parameter that is constant in (jumpCutoff,1-jumpCutoff) and linear to zero
    on either side.'''

    if P >= jumpCutoff and P <= 1-jumpCutoff:
        param = midParam
    elif P < jumpCutoff:
        param = P * midParam / jumpCutoff
    else:
        param = (1-P) * midParam / jumpCutoff

    if param == 0: param = 1e-4
    return param

def _doubleExpCum( J, lam ):
    '''Cumulative distribution for double exponential jump distribution, integrating the
    density from -infinity to J.'''

    if J < 0:
        return 0.5*math.exp( lam * J )
    else:
        return 1 - 0.5 * math.exp( -lam * J )

def _doubleExpInt( J, lam ):
    '''Integral of J f(J) from J=-infinity to the supplied J.'''

    if J < 0:
        return -0.5 * ( 1./lam - J ) * math.exp( lam * J )
    else:
        return -0.5 * ( 1./lam + J ) * math.exp( -lam * J )

def cubefulEquities( W, L, alpha, jumpCutoff, N, tol=0.001, iterMax=8, distrib='normal', verbose=False ):
    '''Calculates equities for three states: centered-cube, player-owned, and opponent-owned.
    A model where jumps are normally distributed with mean zero and a jump volatility that goes linearly
    to zero at P=0% and P=100%, and is constant at alpha inside (jumpCutoff,100%-jumpCutoff).
    "Jump volatility" means expected value of the absolute value of a jump.
    P is the cubeless probably of player winning the game. N is the number of grid points in
    the probability-of-win direction, running from P=0 to P=1. tol is the tolerance applied to
    solving for the cube decision points in the iteration - when all points are converged within
    tol the iteration stops. iterMax is the max number of iterations that will happen, regardless
    of convergence.
    distrib represents the name of the distribution used for jumps. 'normal' and 'doubleexp' are
    allowed.'''

    # if we know six cutoff points in P (opponent TG, TP, opponent RD, player RD, CP, player TG)
    # then we can solve for the player-owned and opponent-owned cube equities by solving a linear
    # system. We don't know these, though, so we'll guess and then iterate until the solution to
    # the linear system is consistent with those points.

    # make a guess at the points by assuming a piecewise linear form for player-owned cube equity
    # in P=(0,CP) then P=(CP,1). Similar for opponent.

    TP  = (L-0.5)*(1+alpha/2.)/(W+L+0.5)
    CP  = (L+1-alpha/2.*(W-0.5))/(W+L+0.5)
    RDp = (L+1-alpha/2.*(L+2*W))/(W+L+0.5)
    RDo = (L-0.5+alpha/2.*(W+2*L))/(W+L+0.5)
    TGp = (4*(1+L)*(W-1)+alpha/2.*(-5+4*L*L+14*W+12*L*W)+alpha*alpha/4.*(-2+6*W-4*W*W))/(1+2*L+2*W)/(2*(W-1)+alpha/2.*(-1+2*L+4*W))
    TGo = (1+alpha/2.)*(1+alpha/2.)*(1-3*L+2*L*L)/(W+L+0.5)/(2*(L-1)+alpha/2.*(-1+4*L+2*W))

    # set up the grid we'll solve for player-owned cube and opponent-owned cube equities on. Discrete in P.

    dP = 1./N
    Ps = [ i*dP for i in xrange(0,N+1) ]

    # iterate until we converge or run through iterMax iterations

    sigma = alpha * math.sqrt( math.pi / 2. )
    
    for iter in xrange(0,iterMax):
        if verbose:
            print 'Iteration', iter, 'cutoff points:'
            print TGo, TP, RDo, RDp, CP, TGp
            print

        m = scipy.matrix( scipy.zeros( (2*N-2,2*N-2) ) )
        b = scipy.matrix( scipy.zeros( (2*N-2,1) ) )

        for i in xrange(0,N-1):
            # calculate the jump volatility of the jump distribution at this point

            alphai = _interpParam( Ps[i+1], jumpCutoff, alpha )

            if distrib=='normal':
                sigmai = alphai * math.sqrt( math.pi / 2. )
            elif distrib=='doubleexp':
                lami = 1./alphai
            else:
                raise ValueError("distrib must be normal or doubleexp")

            # calculate some numbers we'll find useful

            if distrib=='normal':
                k = sigmai/math.sqrt(2*math.pi)

                dNs  = [ norm.cdf( ( Ps[j] - Ps[i+1] ) / sigmai ) - norm.cdf( ( Ps[j-1] - Ps[i+1] ) / sigmai ) for j in xrange(1,N+1) ]
                des  = [ k * ( math.exp( -(Ps[j-1]-Ps[i+1])**2/2./sigmai/sigmai ) - math.exp( -(Ps[j]-Ps[i+1])**2/2./sigmai/sigmai ) ) for j in xrange(1,N+1) ]
            else:
                dNs  = [ _doubleExpCum( Ps[j]-Ps[i+1], lami ) - _doubleExpCum( Ps[j-1]-Ps[i+1], lami ) for j in xrange(1,N+1) ]
                des  = [ _doubleExpInt( Ps[j]-Ps[i+1], lami ) - _doubleExpInt( Ps[j-1]-Ps[i+1], lami ) for j in xrange(1,N+1) ]

            # attribute all the bits

            for j in xrange( 0, N ):
                wjp1 = dNs[j]*(Ps[i+1]-Ps[j])/dP + des[j]/dP
                wj   = -dNs[j]*(Ps[i+1]-Ps[j+1])/dP - des[j]/dP

                # do the bits for Ep

                if j == N-1:
                    b[i,0] -= W*wjp1
                elif Ps[j+1] < RDp or Ps[j+1] > TGp:
                    m[i,j] += wjp1
                elif Ps[j+1] < CP:
                    m[i,j+1+N-2] += wjp1*2
                else:
                    b[i,0] -= wjp1

                if j == 0:
                    b[i,0] += L*wj
                elif Ps[j] < RDp or Ps[j] > TGp:
                    m[i,j-1] += wj
                elif Ps[j] < CP:
                    m[i,j+N-2] += wj*2
                else:
                    b[i,0] -= wj

                # do the bits for Eo

                if j == N-1:
                    b[i+N-1,0] -= W*wjp1
                elif Ps[j+1] < TGo or Ps[j+1] > RDo:
                    m[i+N-1,j+1+N-2] += wjp1
                elif Ps[j+1] < TP:
                    b[i+N-1,0] += wjp1
                else:
                    m[i+N-1,j] += wjp1*2

                if j == 0:
                    b[i+N-1,0] += L*wj
                elif Ps[j] < TGo or Ps[j] > RDo:
                    m[i+N-1,j+N-2] += wj
                elif Ps[j] < TP:
                    b[i+N-1,0] += wj
                else:
                    m[i+N-1,j-1] += wj*2

            m[i,i] -= 1
            m[i+N-1,i+N-1] -= 1

            if distrib=='normal':
                N0i = norm.cdf( (Ps[0]-Ps[i+1])/sigmai )
                e0i = -k*math.exp( -(Ps[0]-Ps[i+1])**2/2./sigmai/sigmai )
                Nni = norm.cdf( (Ps[N]-Ps[i+1])/sigmai )
                eni = -k*math.exp( -(Ps[N]-Ps[i+1])**2/2./sigmai/sigmai )
            else:
                N0i = _doubleExpCum( Ps[0]-Ps[i+1], lami )
                e0i = _doubleExpInt( Ps[0]-Ps[i+1], lami )
                Nni = _doubleExpCum( Ps[N]-Ps[i+1], lami )
                eni = _doubleExpInt( Ps[N]-Ps[i+1], lami )

            b[i,0] += L/dP*(-e0i + (Ps[1]-Ps[i+1])*N0i) - W/dP*(-eni+(Ps[i+1]-Ps[N-1])*(1-Nni))
            b[i+N-1,0] += L/dP*(-e0i + (Ps[1]-Ps[i+1])*N0i) - W/dP*(-eni+(Ps[i+1]-Ps[N-1])*(1-Nni))

            m[i,0] += 1./dP*(e0i+(Ps[i+1]-Ps[0])*N0i)
            m[i,N-2] += 1./dP*(eni+(Ps[N]-Ps[i+1])*(1-Nni))
            m[i+N-1,N-1] += 1./dP*(e0i+(Ps[i+1]-Ps[0])*N0i)
            m[i+N-1,2*N-3] += 1./dP*(eni+(Ps[N]-Ps[i+1])*(1-Nni))

        # calculate the solution to the linear system (with the input values of the six cutoff points)

        Evec = m.I*b

        Eps = [ Evec[i,0] for i in xrange(0,N-1) ]
        Eos = [ Evec[i+N-1,0] for i in xrange(0,N-1) ]

        # now we want to check where the solution finds cutoff points; use those for the next iteration

        TGoNew = _root( Eos, Ps[1:-1], -1 )
        TPNew  = _root( Eps, Ps[1:-1], -0.5 )
        CPNew  = _root( Eos, Ps[1:-1], 0.5 )
        TGpNew = _root( Eps, Ps[1:-1], 1 )

        diffs = [ 2*Eo-Ep for Eo, Ep in zip( Eos, Eps ) ]
        RDpNew = _root( diffs, Ps[1:-1], 0 )
        diffs = [ 2*Ep-Eo for Eo, Ep in zip( Eos, Eps ) ]
        RDoNew = _root( diffs, Ps[1:-1], 0 )

        dTGo = abs(TGoNew-TGo)
        dTP  = abs(TPNew-TP)
        dRDo = abs(RDoNew-RDo)
        dRDp = abs(RDpNew-RDp)
        dCP  = abs(CPNew-CP)
        dTGp = abs(TGpNew-TGp)

        TGo = TGoNew
        TP  = TPNew
        RDo = RDoNew
        RDp = RDpNew
        CP  = CPNew
        TGp = TGpNew

        if dTGo < tol and dTP < tol and dRDo < tol and dRDp < tol and dCP < tol and dTGp < tol:
            break

    # now that we've calculated the equity for cases after the cube has been passed, calculate the
    # centered-cube equity. This introduces four more cutoff points: IDp and IDo, the initial double
    # points for the player and opponent; and TGpc and TGoc, the too-good points for centered cube.

    # our estimates for initial double points will be redouble points, and estimates for centered-cube
    # too-good points will be the cube-owned ones. Should be close enough to get good convergence.

    IDp  = RDp
    IDo  = RDo
    TGpc = TGp
    TGoc = TGo

    for iter in xrange( 0, iterMax ):
        if verbose:
            print 'Centered cube iteration', iter, 'cutoff points:'
            print TGoc, IDo, IDp, TGpc
            print

        m = scipy.matrix( scipy.zeros( (N-1,N-1) ) )
        b = scipy.matrix( scipy.zeros( (N-1,1) ) )

        for i in xrange(0,N-1):
            # calculate the jump volatility of the jump distribution at this point

            alphai = _interpParam( Ps[i+1], jumpCutoff, alpha )

            if distrib=='normal':
                sigmai = alphai * math.sqrt( math.pi / 2. )
            else:
                lami = 1./alphai

            # calculate some numbers we'll find useful

            if distrib=='normal':
                k = sigmai/math.sqrt(2*math.pi)

                dNs  = [ norm.cdf( ( Ps[j] - Ps[i+1] ) / sigmai ) - norm.cdf( ( Ps[j-1] - Ps[i+1] ) / sigmai ) for j in xrange(1,N+1) ]
                des  = [ k * ( math.exp( -(Ps[j-1]-Ps[i+1])**2/2./sigmai/sigmai ) - math.exp( -(Ps[j]-Ps[i+1])**2/2./sigmai/sigmai ) ) for j in xrange(1,N+1) ]
            else:
                dNs  = [ _doubleExpCum( Ps[j]-Ps[i+1], lami ) - _doubleExpCum( Ps[j-1]-Ps[i+1], lami ) for j in xrange(1,N+1) ]
                des  = [ _doubleExpInt( Ps[j]-Ps[i+1], lami ) - _doubleExpInt( Ps[j-1]-Ps[i+1], lami ) for j in xrange(1,N+1) ]

            # attribute all the bits

            for j in xrange( 0, N ):
                wjp1 = dNs[j]*(Ps[i+1]-Ps[j])/dP + des[j]/dP
                wj   = -dNs[j]*(Ps[i+1]-Ps[j+1])/dP - des[j]/dP

                if j == N-1:
                    b[i,0] -= W*wjp1
                elif Ps[j+1] < TGoc or Ps[j+1] > TGpc:
                    m[i,j] += wjp1
                elif Ps[j+1] < TP:
                    b[i,0] += wjp1
                elif Ps[j+1] < IDo:
                    b[i,0] -= 2*Eps[j]*wjp1
                elif Ps[j+1] < IDp:
                    m[i,j] += wjp1
                elif Ps[j+1] < CP:
                    b[i,0] -= 2*Eos[j]*wjp1
                else:
                    b[i,0] -= wjp1

                if j == 0:
                    b[i,0] += L*wj
                elif Ps[j] < TGoc or Ps[j] > TGpc:
                    m[i,j-1] += wj
                elif Ps[j] < TP:
                    b[i,0] += wj
                elif Ps[j] < IDo:
                    b[i,0] -= 2*Eps[j-1]*wj
                elif Ps[j] < IDp:
                    m[i,j-1] += wj
                elif Ps[j] < CP:
                    b[i,0] -= 2*Eos[j-1]*wj
                else:
                    b[i,0] -= wj

            m[i,i] -= 1

            if distrib=='normal':
                N0i = norm.cdf( (Ps[0]-Ps[i+1])/sigmai )
                e0i = -k*math.exp( -(Ps[0]-Ps[i+1])**2/2./sigmai/sigmai )
                Nni = norm.cdf( (Ps[N]-Ps[i+1])/sigmai )
                eni = -k*math.exp( -(Ps[N]-Ps[i+1])**2/2./sigmai/sigmai )
            else:
                N0i = _doubleExpCum( Ps[0]-Ps[i+1], lami )
                e0i = _doubleExpInt( Ps[0]-Ps[i+1], lami )
                Nni = _doubleExpCum( Ps[N]-Ps[i+1], lami )
                eni = _doubleExpInt( Ps[N]-Ps[i+1], lami )

            b[i,0] += L/dP*(-e0i + (Ps[1]-Ps[i+1])*N0i) - W/dP*(-eni+(Ps[i+1]-Ps[N-1])*(1-Nni))

            m[i,0] += 1./dP*(e0i+(Ps[i+1]-Ps[0])*N0i)
            m[i,N-2] += 1./dP*(eni+(Ps[N]-Ps[i+1])*(1-Nni))

        # calculate the solution to the linear system

        Evec = m.I*b
        Ecs  = [ Evec[i,0] for i in xrange(0,N-1) ]

        # now iterate on the four points

        TGocNew = _root( Ecs, Ps[1:-1], -1 )
        TGpcNew = _root( Ecs, Ps[1:-1], 1 )
        diffs = [ 2*Eo - Ec for Eo, Ec in zip(Eos,Ecs) ]
        IDpNew  = _root( diffs, Ps[1:-1], 0 )
        diffs = [ 2*Ep - Ec for Ep, Ec in zip(Eps,Ecs) ]
        IDoNew  = _root( diffs, Ps[1:-1], 0 )

        dTGoc = abs(TGocNew-TGoc)
        dIDo  = abs(IDoNew-IDo)
        dIDp  = abs(IDpNew-IDp)
        dTGpc = abs(TGpcNew-TGpc)

        TGoc = TGocNew
        IDo  = IDoNew
        IDp  = IDpNew
        TGpc = TGpcNew

        if dTGoc < tol and dIDo < tol and dIDp < tol and dTGpc < tol:
            break

    # add the first and last points to all the equity vectors

    Ecs = [ -L ] + Ecs + [ W ]
    Eps = [ -L ] + Eps + [ W ]
    Eos = [ -L ] + Eos + [ W ]

    return Ps, Ecs, Eps, Eos, TGoc, TGo, TP, RDo, IDo, IDp, RDp, CP, TGp, TGpc

def cubefulEquityApproxLinear( P, W, L, alpha, jumpCutoff, cube, playerOwnsCube ):
    '''Approximate cubeful equity for the given cube level and cube ownership. Not normalized
    by cube value.'''

    # get the live cube limit take and cash points - used later

    TPl = (L-0.5)/(W+L+0.5)
    CPl = (L+1)/(W+L+0.5)

    # calculate the model take and cash points
    
    TP  = TPl*(L+1)/(L+1-alpha/4.*(W+L+0.5)/(W-0.5))
    CP  = CPl-alpha*(W-0.5)/(2*(2*L*W+2*L-W-1)-alpha*(W+L+0.5))

    # define the parameters of the different piecewise linear functions. I assume
    # equity = A + B P, and for each equity (player-owned, unavailable, and
    # centered-cube) there are multiple pieces. "o" subscripts mean player-owned;
    # "u" means unavailable; and "c" means centered.

    A1o = -L
    B1o = W+L+0.5 - alpha/4.*(W+L+0.5)**2/(W-0.5)/(L+1)
    ECP = A1o + B1o*CP

    B2u = W+L+0.5 - alpha/4.*(W+L+0.5)**2/(L-0.5)/(W+1)
    A2u = W-B2u
    ETP = A2u + B2u*TP
    
    B2o = (W-ECP)/(1-CP)
    A2o = W - B2o

    A1u = -L
    B1u = (ETP+L)/TP

    Eh = 1-alpha/6.*(W+L+0.5)*(W+1)/(W-0.5)
    El = -1+alpha/6.*(W+L+0.5)*(L+1)/(L-0.5)
    B2c = (Eh-El)/(CPl-TPl)
    A2c = Eh-B2c*CPl

    ECP = A2c + B2c*CP
    ETP = A2c + B2c*TP
    
    A1c = -L
    B1c = (ETP+L)/TP
    B3c = (W-ECP)/(1-CP)
    A3c = W-B3c

    # return the appropriate cube-state equity for the appropriate piece of P-space

    if cube == 1:
        if P<TP:
            return A1c + B1c*P
        elif P<CP:
            return A2c + B2c*P
        else:
            return A3c + B3c*P
    elif playerOwnsCube:
        if P<CP:
            return cube*(A1o + B1o*P)
        else:
            return cube*(A2o + B2o*P)
    else:
        if P<TP:
            return cube*(A1u + B1u*P)
        else:
            return cube*(A2u + B2u*P)

def cubefulEquityApprox( P, W, L, alpha, jumpCutoff, cube, playerOwnsCube ):
    '''Approximate cubeful equity for the given cube level and cube ownership. Not normalized
    by cube value.'''

    # get the live cube limit take and cash points - used later

    TPl = (L-0.5)/(W+L+0.5)
    CPl = (L+1)/(W+L+0.5)

    # calculate the model take and cash points
    
    TP  = TPl*(L+1)/(L+1-alpha/4.*(W+L+0.5)/(W-0.5))
    CP  = CPl-alpha*(W-0.5)/(2*(2*L*W+2*L-W-1)-alpha*(W+L+0.5))

    # define the parameters of the different piecewise linear functions. I assume
    # equity = A + B P, and for each equity (player-owned, unavailable, and
    # centered-cube) there are multiple pieces. "o" subscripts mean player-owned;
    # "u" means unavailable; and "c" means centered.

    A1o = -L
    B1o = W+L+0.5 - alpha/4.*(W+L+0.5)**2/(W-0.5)/(L+1)
    ECP = A1o + B1o*CP

    B2u = W+L+0.5 - alpha/4.*(W+L+0.5)**2/(L-0.5)/(W+1)
    A2u = W-B2u
    ETP = A2u + B2u*TP
    
    B2o = (W-ECP)/(1-CP)
    A2o = W - B2o

    A1u = -L
    B1u = (ETP+L)/TP

    Eh = 1-alpha/6.*(W+L+0.5)*(W+1)/(W-0.5)
    El = -1+alpha/6.*(W+L+0.5)*(L+1)/(L-0.5)
    B2c = (Eh-El)/(CPl-TPl)
    A2c = Eh-B2c*CPl

    ECP = A2c + B2c*CP
    ETP = A2c + B2c*TP
    
    A1c = -L
    B1c = (ETP+L)/TP
    B3c = (W-ECP)/(1-CP)
    A3c = W-B3c

    RDo = (A1o-2*A2u)/(2*B2u-B1o)
    RDu = (A2u-2*A1o)/(2*B1o-B2u)
    TGo = (1-A2o)/B2o
    TGu = (-1-A1u)/B1u
    IDo = (A2c-2*A2u)/(2*B2u-B2c)
    IDu = (A2c-2*A1o)/(2*B1o-B2c)

    # use the double-exponential jump distribution to calculate a better estimate of the true
    # cubeful equities by integrating over the linear approximation equities. The choice of
    # distribution here matters a little but not significantly. Since jumps might take us to
    # P<0 or P>1 we approximate the equity in that space as linear as well, with slopes set to
    # force E(P=0) = -L and E(P=1) = +W. This is a hack to deal with the fact that the jump
    # distribution isn't sufficiently complex to avoid jumping into non-physical territory.

    lam = 1./alpha
    
    if cube==1:
        # figure out the slope of equity vs prob we need for P<0 and P>1 such that we get the right
        # values at each boundary

        Al = -L
        Bl = 2./alpha*( L/2. + (A1c)*(_doubleExpCum(TGu,lam)-0.5) + B1c*(_doubleExpInt(TGu,lam)+alpha/2.) \
            + (-1)*( _doubleExpCum(TP,lam) - _doubleExpCum(TGu,lam) ) \
            + 2*(A1o)*( _doubleExpCum(IDu,lam) - _doubleExpCum(TP,lam) ) + 2*B1o*( _doubleExpInt(IDu,lam) - _doubleExpInt(TP,lam) ) \
            + (A2c)*( _doubleExpCum(IDo,lam) - _doubleExpCum(IDu,lam) ) + B2c*( _doubleExpInt(IDo,lam) - _doubleExpInt(IDu,lam) ) \
            + 2*(A2u)*( _doubleExpCum(CP,lam) - _doubleExpCum(IDo,lam) ) + 2*B2u*( _doubleExpInt(CP,lam) - _doubleExpInt(IDo,lam) ) \
            + _doubleExpCum(TGo,lam) - _doubleExpCum(CP,lam) \
            + (A3c)*(1-_doubleExpCum(TGo,lam)) - B3c*_doubleExpInt(TGo,lam) )
        Bh = 2./alpha*( W/2. - ( (A1c+B1c)*_doubleExpCum(TGu-1,lam) + B1c*_doubleExpInt(TGu-1,lam) \
            + (-1)*( _doubleExpCum(TP-1,lam) - _doubleExpCum(TGu-1,lam) ) \
            + 2*(A1o+B1o)*( _doubleExpCum(IDu-1,lam) - _doubleExpCum(TP-1,lam) ) + 2*B1o*( _doubleExpInt(IDu-1,lam) - _doubleExpInt(TP-1,lam) ) \
            + (A2c+B2c)*( _doubleExpCum(IDo-1,lam) - _doubleExpCum(IDu-1,lam) ) + B2c*( _doubleExpInt(IDo-1,lam) - _doubleExpInt(IDu-1,lam) ) \
            + 2*(A2u+B2u)*( _doubleExpCum(CP-1,lam) - _doubleExpCum(IDo-1,lam) ) + 2*B2u*( _doubleExpInt(CP-1,lam) - _doubleExpInt(IDo-1,lam) ) \
            + _doubleExpCum(TGo-1,lam) - _doubleExpCum(CP-1,lam) \
            + (A3c+B3c)*(0.5-_doubleExpCum(TGo-1,lam)) + B3c*(-alpha/2.-_doubleExpInt(TGo-1,lam)) ) )
        Ah = W-Bh
        
        equityNorm = (Al+Bl*P)*_doubleExpCum(-P,lam)+Bl*_doubleExpInt(-P,lam) \
            + (A1c+B1c*P)*(_doubleExpCum(TGu-P,lam)-_doubleExpCum(-P,lam)) + B1c*(_doubleExpInt(TGu-P,lam)-_doubleExpInt(-P,lam)) \
            + (-1)*( _doubleExpCum(TP-P,lam) - _doubleExpCum(TGu-P,lam) ) \
            + 2*(A1o+B1o*P)*( _doubleExpCum(IDu-P,lam) - _doubleExpCum(TP-P,lam) ) + 2*B1o*( _doubleExpInt(IDu-P,lam) - _doubleExpInt(TP-P,lam) ) \
            + (A2c+B2c*P)*( _doubleExpCum(IDo-P,lam) - _doubleExpCum(IDu-P,lam) ) + B2c*( _doubleExpInt(IDo-P,lam) - _doubleExpInt(IDu-P,lam) ) \
            + 2*(A2u+B2u*P)*( _doubleExpCum(CP-P,lam) - _doubleExpCum(IDo-P,lam) ) + 2*B2u*( _doubleExpInt(CP-P,lam) - _doubleExpInt(IDo-P,lam) ) \
            + _doubleExpCum(TGo-P,lam) - _doubleExpCum(CP-P,lam) \
            + (A3c+B3c*P)*(_doubleExpCum(1-P,lam)-_doubleExpCum(TGo-P,lam)) + B3c*(_doubleExpInt(1-P,lam)-_doubleExpInt(TGo-P,lam)) \
            + (Ah+Bh*P)*(1-_doubleExpCum(1-P,lam))-Bh*_doubleExpInt(1-P,lam)
    elif playerOwnsCube:
        # adjust the function at P=1 to make it better hit W at P=1 - find the approximate linear fn for P>1 to make it so

        Bh = 2./alpha*( W/2 - ( (A1o+B1o)*_doubleExpCum(RDo-1,lam) + B1o * _doubleExpInt(RDo-1,lam) \
            + 2*(A2u+B2u)*( _doubleExpCum(CP-1,lam) - _doubleExpCum(RDo-1,lam) ) + 2*B2u*( _doubleExpInt(CP-1,lam) - _doubleExpInt(RDo-1,lam) ) \
            + _doubleExpCum(TGo-1,lam) - _doubleExpCum(CP-1,lam) \
            + (A2o+B2o)*(0.5-_doubleExpCum(TGo-1,lam)) + B2o*(-alpha/2.-_doubleExpInt(TGo-1,lam)) ) )
        Ah = W - Bh
        
        equityNorm = (A1o+B1o*P)*_doubleExpCum(RDo-P,lam) + B1o * _doubleExpInt(RDo-P,lam) \
            + 2*(A2u+B2u*P)*( _doubleExpCum(CP-P,lam) - _doubleExpCum(RDo-P,lam) ) + 2*B2u*( _doubleExpInt(CP-P,lam) - _doubleExpInt(RDo-P,lam) ) \
            + _doubleExpCum(TGo-P,lam) - _doubleExpCum(CP-P,lam) \
            + (A2o+B2o*P)*(_doubleExpCum(1-P,lam)-_doubleExpCum(TGo-P,lam)) + B2o*(_doubleExpInt(1-P,lam)-_doubleExpInt(TGo-P,lam)) \
            + (Ah+Bh*P)*(1-_doubleExpCum(1-P,lam)) - Bh*_doubleExpInt(1-P,lam)
    else:
        # adjust the function at P=0 to make it better hit -L at P=0

        Al = -L
        Bl = 2./alpha*( L/2. + (A1u)*(_doubleExpCum(TGu,lam)-0.5) + B1u*(_doubleExpInt(TGu,lam)+alpha/2.) \
            + (-1)*( _doubleExpCum(TP,lam) - _doubleExpCum(TGu,lam) ) \
            + 2*(A1o)*( _doubleExpCum(RDu,lam) - _doubleExpCum(TP,lam) ) + 2*B1o*( _doubleExpInt(RDu,lam) - _doubleExpInt(TP,lam) ) \
            + (A2u)*( 1 - _doubleExpCum(RDu,lam) ) - B2u*_doubleExpInt(RDu,lam) )
        
        equityNorm = (Al+Bl*P)*_doubleExpCum(-P,lam) + Bl*_doubleExpInt(-P,lam) \
            + (A1u+B1u*P)*(_doubleExpCum(TGu-P,lam)-_doubleExpCum(-P,lam)) + B1u*(_doubleExpInt(TGu-P,lam)-_doubleExpInt(-P,lam)) \
            + (-1)*( _doubleExpCum(TP-P,lam) - _doubleExpCum(TGu-P,lam) ) \
            + 2*(A1o+B1o*P)*( _doubleExpCum(RDu-P,lam) - _doubleExpCum(TP-P,lam) ) + 2*B1o*( _doubleExpInt(RDu-P,lam) - _doubleExpInt(TP-P,lam) ) \
            + (A2u+B2u*P)*( 1 - _doubleExpCum(RDu-P,lam) ) - B2u*_doubleExpInt(RDu-P,lam)
        
    return equityNorm * cube

def test():
    '''Test out the cubeful equity evaluation'''

    # set up the game state parameters

    W = 1.4
    L = 1.
    alpha = 0.1
    jumpCutoff = 0.

    # calculate the numerical estimates of model equities and cube decision points
    
    N   = 100
    res = cubefulEquities( W, L, alpha, jumpCutoff, N, 0.001, 8, 'doubleexp' )
    print 'Cube decision points:'
    print res[4:] # see cubefulEquities fn for order of cube decision points here
    print

    # calculate the linear and nonlinear approximate equities for each P value in
    # the numerical calc and print them out along with the numerical equities.

    for i in xrange(0,len(res[0])):
        P  = res[0][i]
        Eo = res[2][i]
        Eu = res[3][i]
        Ec = res[1][i]

        # get the nonlinear approximation equities (normalized by the cube value)

        Eca = cubefulEquityApprox( P, W, L, alpha, jumpCutoff, 1, True )
        Eoa = cubefulEquityApprox( P, W, L, alpha, jumpCutoff, 2, True ) / 2.
        Eua = cubefulEquityApprox( P, W, L, alpha, jumpCutoff, 2, False ) / 2.

        # get the linear approximation equities (normalized)
        
        Ecal = cubefulEquityApproxLinear( P, W, L, alpha, jumpCutoff, 1, True )
        Eoal = cubefulEquityApproxLinear( P, W, L, alpha, jumpCutoff, 2, True ) / 2.
        Eual = cubefulEquityApproxLinear( P, W, L, alpha, jumpCutoff, 2, False ) / 2.

        # print it out as a comma-delimited list. Easy to c&p into Excel, e.g.
        
        print str(P)  + "," + str(Ec)  + ',' + str(Eca)  + ',' + str(Ecal)  + ',' \
            + str(Eo) + ',' + str(Eoa) + ',' + str(Eoal) + ',' \
            + str(Eu) + ',' + str(Eua) + ',' + str(Eual) 
    
    return res

def test2():
    '''Test out diffs btw distributional assumptions'''

    W = 1
    L = 1
    alpha = 0.4
    jumpCutoff = 0.
    N = 100

    # potentially different alphas for the two to examine the impact

    alpha1 = alpha
    alpha2 = alpha
    #alpha2 = math.sqrt(math.pi/4)*alpha
    #alpha2 = 1/math.sqrt(2)*alpha

    res1 = cubefulEquities( W, L, alpha1, jumpCutoff, N, 0.001, 8, 'normal' )
    res2 = cubefulEquities( W, L, alpha2, jumpCutoff, N, 0.001, 8, 'doubleexp' )
    print res1[4:]
    print res2[4:]

    for i in xrange(0,len(res1[0])):
        print str(res1[0][i]) + ',' + str(res1[1][i]) + ',' + str(res2[1][i]) + ',' + str(res1[1][i]-res2[1][i])


def test3():
    '''demonstrate that the model equity is between the dead and live cube limits.'''

    W = 1.25
    L = 1.25
    alpha = 0.1

    N = 100
    dP = 1./N

    for i in xrange(0,N+1):
        P = i*dP
        Em = cubefulEquityApprox(P,W,L,alpha,0,2,True)/2
        Ed = P*(W+L)-L
        El = cubefulEquityApproxLinear(P,W,L,0,0,2,True)/2

        print str(P)+","+str(Em)+","+str(Ed)+","+str(El)
    
def test4():
    '''Check cube decision points btw linear and nonlinear approximations.'''

    W = 1.25
    L = 1.25
    alpha = 0.113

    print 'Alpha =', alpha

    def argFunc(P):
        return cubefulEquityApprox(P,W,L,alpha,0,2,True)+1

    TPest = (L-0.5)/(L+W+0.5)*(L+1)/(L+1-alpha/4.*(W+L+0.5)/(W-0.5))

    TP = scipy.optimize.brenth(argFunc,TPest-0.01,TPest+0.01)

    print 'TP linear =', TPest
    print 'TP nonl   =', TP

    def argFunc(P):
        return cubefulEquityApprox(P,W,L,alpha,0,2,False)-1
    
    CPest = (L+1)/(W+L+0.5) - alpha*(W-0.5)/(2*(2*L*W+2*L-W-1)-alpha*(W+L+0.5))

    CP = scipy.optimize.brenth(argFunc,CPest-0.01,CPest+0.01)

    print 'CP linear =', CPest
    print 'CP nonl   =', CP

    def argFunc(P):
        return cubefulEquityApproxLinear(P,W,L,alpha,0,2,True)-cubefulEquityApproxLinear(P,W,L,alpha,0,4,False)

    RDest = scipy.optimize.brenth(argFunc,TPest,CPest)

    def argFunc(P):
        return cubefulEquityApprox(P,W,L,alpha,0,2,True)-cubefulEquityApprox(P,W,L,alpha,0,4,False)

    RD = scipy.optimize.brenth(argFunc,TP,CP)

    print 'RD linear =', RDest
    print 'RD nonl   =', RD

    def argFunc(alphaNL):
        def intFunc(P):
            return cubefulEquityApprox(P,W,L,alphaNL,0,2,True)-cubefulEquityApprox(P,W,L,alphaNL,0,4,False)
            
        RD = scipy.optimize.brenth(intFunc,TP,CP)
        return RD-RDest

    alphaNL = scipy.optimize.brenth(argFunc,alpha-0.05,alpha+0.05)
    print 'Eff alpha =', alphaNL
    print 'Diff      =', alphaNL - alpha
    print
    
    def argFunc(P):
        return cubefulEquityApproxLinear(P,W,L,alpha,0,2,True)-2
        
    TGest = scipy.optimize.brenth(argFunc,CPest,1)

    def argFunc(P):
        return cubefulEquityApprox(P,W,L,alpha,0,2,True)-2

    TG = scipy.optimize.brenth(argFunc,CP,1)

    print 'TG linear =', TGest
    print 'TG nonl   =', TG

    def argFunc(alphaNL):
        def intFunc(P):
            return cubefulEquityApprox(P,W,L,alphaNL,0,2,True)-2
            
        TG = scipy.optimize.brenth(intFunc,CP,1)
        return TG-TGest

    alphaNL = scipy.optimize.brenth(argFunc,alpha-0.05,alpha+0.05)
    print 'Eff alpha =', alphaNL
    print 'Diff      =', alphaNL - alpha
    print

    def argFunc(P):
        return cubefulEquityApproxLinear(P,W,L,alpha,0,1,False)-cubefulEquityApproxLinear(P,W,L,alpha,0,2,False)
        
    IDest = scipy.optimize.brenth(argFunc,TPest,RDest)

    def argFunc(P):
        return cubefulEquityApprox(P,W,L,alpha,0,1,False)-cubefulEquityApprox(P,W,L,alpha,0,2,False)

    ID = scipy.optimize.brenth(argFunc,TP,RD)

    print 'ID linear =', IDest
    print 'ID nonl   =', ID

    def argFunc(alphaNL):
        def intFunc(P):
            return cubefulEquityApprox(P,W,L,alphaNL,0,1,False)-cubefulEquityApprox(P,W,L,alphaNL,0,2,False)
            
        ID = scipy.optimize.brenth(intFunc,TP,CP)
        return ID - IDest
    
    alphaNL = scipy.optimize.brenth(argFunc,alpha-0.05,alpha+0.05)
    print 'Eff alpha =', alphaNL
    print 'Diff      =', alphaNL - alpha
    print

