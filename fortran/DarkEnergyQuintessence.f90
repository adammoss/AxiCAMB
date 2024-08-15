    ! Equations module allowing for fairly general quintessence models
    !
    ! by Antony Lewis (http://cosmologist.info/)

    !!FIX March 2005: corrected update to next treatment of tight coupling
    !!Fix Oct 2011: a^2 factor in ayprime(EV%w_ix+1) [thanks Raphael Flauger]
    ! Oct 2013: update for latest CAMB, thanks Nelson Lima, Martina Schwind
    ! May 2020: updated for CAMB 1.x+

    ! Notes at http://antonylewis.com/notes/CAMB.pdf

    !This module is not well tested, use at your own risk!

    !Need to specify Vofphi function, and also initial_phi
    !You may also need to change other things to get it to work with different types of quintessence model

    !It works backwards, in that it assumes Omega_de is Omega_Q today, then does a binary search on the
    !initial conditions to find what is required to give that Omega_Q today after evolution.

    module Quintessence
    use DarkEnergyInterface
    use results
    use constants
    use classes
    use Interpolation
    implicit none
    private

    real(dl), parameter :: Tpl= sqrt(kappa*hbar/c**5)  ! sqrt(8 pi G hbar/c^5), reduced planck time

    ! General base class. Specific implemenetations should inherit, defining Vofphi and setting up
    ! initial conditions and interpolation tables
    type, extends(TDarkEnergyModel) :: TQuintessence
        integer :: DebugLevel = 1 !higher then zero for some debug output to console
        real(dl) :: astart = 1e-7_dl
        real(dl) :: integrate_tol = 1e-6_dl
        real(dl), dimension(:), allocatable :: sampled_a, phi_a, phidot_a
        ! Steps for log a and linear spacing, switching at max_a_log (set by Init)
        integer, private :: npoints_linear, npoints_log
        real(dl), private :: dloga, da, log_astart, max_a_log
        real(dl), private, dimension(:), allocatable :: ddphi_a, ddphidot_a
        class(CAMBdata), pointer, private :: State
    contains
    procedure :: Vofphi !V(phi) potential [+ any cosmological constant]
    procedure :: ValsAta !get phi and phi' at scale factor a, e.g. by interpolation in precomputed table
    procedure :: Init => TQuintessence_Init
    procedure :: PerturbedStressEnergy => TQuintessence_PerturbedStressEnergy
    procedure :: PerturbationEvolve => TQuintessence_PerturbationEvolve
    procedure :: BackgroundDensityAndPressure => TQuintessence_BackgroundDensityAndPressure
    procedure :: EvolveBackground
    procedure :: EvolveBackgroundLog
    procedure, private :: phidot_start => TQuintessence_phidot_start
    end type TQuintessence

    ! Specific implementation for early quintessence + cosmologial constant, assuming the early component
    ! energy density fraction is negligible at z=0.
    ! The specific parameterization of the potential implemented is the axion model of arXiv:1908.06995
    type, extends(TQuintessence) :: TEarlyQuintessence
        real(dl) :: n = 3._dl
        real(dl) :: f =0.05 ! sqrt(8*pi*G)*f
        real(dl) :: m = 5d-54 !m in reduced Planck mass units
        real(dl) :: theta_i = 3.1_dl !initial value of phi/f
        real(dl) :: frac_lambda0 = 1._dl !fraction of dark energy density that is cosmological constant today
        logical :: use_zc = .true. !adjust m to fit zc
        real(dl) :: zc, fde_zc !readshift for peak f_de and f_de at that redshift
        integer :: oscillation_threshold = 1
        logical :: use_fluid_approximation = .false. 
        logicAL :: use_PH = .false.
        integer :: npoints = 5000 !baseline number of log a steps; will be increased if needed when there are oscillations
        integer :: min_steps_per_osc = 10
        real(dl), dimension(:), allocatable :: fde, ddfde
        real(dl), dimension(:), allocatable :: sampled_a_fluid, grhov_fluid, ddgrhov_fluid, w_fluid, ddw_fluid, dwdloga_fluid, dddwdloga_fluid
        integer, private :: npoints_fluid
        real(dl), private :: dloga_fluid
        real(dl), private :: a_fluid_switch = 1._dl
    contains
    procedure :: Vofphi => TEarlyQuintessence_VofPhi
    procedure :: BackgroundDensityAndPressure => TEarlyQuintessence_BackgroundDensityAndPressure
    procedure :: PerturbationEvolve => TEarlyQuintessence_PerturbationEvolve
    procedure :: PerturbedStressEnergy => TEarlyQuintessence_PerturbedStressEnergy
    procedure :: Init => TEarlyQuintessence_Init
    procedure :: ReadParams =>  TEarlyQuintessence_ReadParams
    procedure, nopass :: PythonClass => TEarlyQuintessence_PythonClass
    procedure, nopass :: SelfPointer => TEarlyQuintessence_SelfPointer
    procedure, private :: fdeAta
    procedure, private :: fde_peak
    procedure, private :: check_error
    procedure :: calc_zc_fde
    procedure :: FluidValsAta
    procedure :: has_switch => TEarlyQuintessence_has_switch
    procedure :: Switch => TEarlyQuintessence_Switch
    procedure :: calc_auxillary

    end type TEarlyQuintessence

    procedure(TClassDverk) :: dverk

    public TQuintessence, TEarlyQuintessence
    contains

    function VofPhi(this, phi, deriv, component)
    !Get the quintessence potential as function of phi
    !The input variable phi is sqrt(8*Pi*G)*psi, where psi is the field
    !Returns (8*Pi*G)^(1-deriv/2)*d^{deriv}V(psi)/d^{deriv}psi evaluated at psi
    !return result is in 1/Mpc^2 units [so times (Mpc/c)^2 to get units in 1/Mpc^2]
    class(TQuintessence) :: this
    real(dl) phi,Vofphi
    integer deriv
    integer, intent(in), optional :: component

    call MpiStop('Quintessence classes must override to provide VofPhi')
    VofPhi = 0
    !if (deriv==0) then
    !    Vofphi= norm*this%m*exp(-this%sigma_model*phi)
    !else if (deriv ==1) then
    !    Vofphi=-norm*this%m*sigma_model*exp(-this%sigma_model*phi)
    !else if (deriv ==2) then
    !    Vofphi=norm*this%m*sigma_model**2*exp(-this%sigma_model*phi)
    !else
    !    stop 'Invalid deriv in Vofphi'
    !end if
    !VofPhi = VOfPhi* MPC_in_sec**2 /Tpl**2  !convert to units of 1/Mpc^2


    end function VofPhi


    subroutine TQuintessence_Init(this, State)
    class(TQuintessence), intent(inout) :: this
    class(TCAMBdata), intent(in), target :: State

    !Make interpolation table, etc,
    !At this point massive neutrinos have been initialized
    !so grho_no_de can be used to get density and pressure of other components at scale factor a

    select type(State)
    class is (CAMBdata)
        this%State => State
    end select

    this%is_cosmological_constant = .false.
    this%num_perturb_equations = 2

    this%log_astart = log(this%astart)

    end subroutine  TQuintessence_Init

    subroutine TQuintessence_BackgroundDensityAndPressure(this, grhov, a, grhov_t, w)
    !Get grhov_t = 8*pi*rho_de*a**2 and (optionally) equation of state at scale factor a
    class(TQuintessence), intent(inout) :: this
    real(dl), intent(in) :: grhov, a
    real(dl), intent(out) :: grhov_t
    real(dl), optional, intent(out) :: w
    real(dl) V, a2, grhov_lambda, phi, phidot

    if (this%is_cosmological_constant) then
        grhov_t = grhov * a * a
        if (present(w)) w = -1_dl
    elseif (a >= this%astart) then
        a2 = a**2
        call this%ValsAta(a,phi,phidot)
        V = this%Vofphi(phi,0)
        grhov_t = phidot**2/2 + a2*V
        if (present(w)) then
            w = (phidot**2/2 - a2*V)/grhov_t
        end if
    else
        grhov_t=0
        if (present(w)) w = -1
    end if

    end subroutine TQuintessence_BackgroundDensityAndPressure

    subroutine EvolveBackgroundLog(this,num,loga,y,yprime)
    ! Evolve the background equation in terms of loga.
    ! Variables are phi=y(1), a^2 phi' = y(2)
    ! Assume otherwise standard background components
    class(TQuintessence) :: this
    integer num
    real(dl) y(num),yprime(num)
    real(dl) loga, a

    a = exp(loga)
    call this%EvolveBackground(num, a, y, yprime)
    yprime = yprime*a

    end subroutine EvolveBackgroundLog

    subroutine EvolveBackground(this,num,a,y,yprime)
    ! Evolve the background equation in terms of a.
    ! Variables are phi=y(1), a^2 phi' = y(2)
    ! Assume otherwise standard background components
    class(TQuintessence) :: this
    integer num
    real(dl) y(num),yprime(num)
    real(dl) a, a2, tot
    real(dl) phi, grhode, phidot, adot

    a2=a**2
    phi = y(1)
    phidot = y(2)/a2

    grhode=a2*(0.5d0*phidot**2 + a2*this%Vofphi(phi,0))
    tot = this%state%grho_no_de(a) + grhode

    adot=sqrt(tot/3.0d0)
    yprime(1)=phidot/adot !d phi /d a
    yprime(2)= -a2**2*this%Vofphi(phi,1)/adot

    end subroutine EvolveBackground


    real(dl) function TQuintessence_phidot_start(this,phi)
    class(TQuintessence) :: this
    real(dl) :: phi

    TQuintessence_phidot_start = 0

    end function TQuintessence_phidot_start

    subroutine ValsAta(this,a,aphi,aphidot)
    class(TQuintessence) :: this
    !Do interpolation for background phi and phidot at a (precomputed in Init)
    real(dl) a, aphi, aphidot
    real(dl) a0,b0,ho2o6,delta,da
    integer ix

    if (a >= 0.9999999d0) then
        aphi= this%phi_a(this%npoints_linear+this%npoints_log)
        aphidot= this%phidot_a(this%npoints_linear+this%npoints_log)
        return
    elseif (a < this%astart) then
        aphi = this%phi_a(1)
        aphidot = 0
        return
    elseif (a > this%max_a_log) then
        delta= a-this%max_a_log
        ix = this%npoints_log + int(delta/this%da)
    else
        delta= log(a)-this%log_astart
        ix = int(delta/this%dloga)+1
    end if
    da = this%sampled_a(ix+1) - this%sampled_a(ix)
    a0 = (this%sampled_a(ix+1) - a)/da
    b0 = 1 - a0
    ho2o6 = da**2/6._dl
    aphi=b0*this%phi_a(ix+1) + a0*(this%phi_a(ix)-b0*((a0+1)*this%ddphi_a(ix)+(2-a0)*this%ddphi_a(ix+1))*ho2o6)
    aphidot=b0*this%phidot_a(ix+1) + a0*(this%phidot_a(ix)-b0*((a0+1)*this%ddphidot_a(ix)+(2-a0)*this%ddphidot_a(ix+1))*ho2o6)

    end subroutine ValsAta

    subroutine TQuintessence_PerturbedStressEnergy(this, dgrhoe, dgqe, &
        a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1, ay, ayprime, w_ix)
    !Get density perturbation and heat flux
    class(TQuintessence), intent(inout) :: this
    real(dl), intent(out) :: dgrhoe, dgqe
    real(dl), intent(in) ::  a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1
    real(dl), intent(in) :: ay(*)
    real(dl), intent(inout) :: ayprime(*)
    integer, intent(in) :: w_ix
    real(dl) phi, phidot, clxq, vq

    call this%ValsAta(a,phi,phidot)
    clxq=ay(w_ix)
    vq=ay(w_ix+1)
    dgrhoe= phidot*vq +clxq*a**2*this%Vofphi(phi,1)
    dgqe= k*phidot*clxq

    end subroutine TQuintessence_PerturbedStressEnergy


    subroutine TQuintessence_PerturbationEvolve(this, ayprime, w, w_ix, &
        a, adotoa, k, z, y)
    !Get conformal time derivatives of the density perturbation and velocity
    class(TQuintessence), intent(in) :: this
    real(dl), intent(inout) :: ayprime(:)
    real(dl), intent(in) :: a, adotoa, w, k, z, y(:)
    integer, intent(in) :: w_ix
    real(dl) clxq, vq, phi, phidot

    call this%ValsAta(a,phi,phidot) !wasting time calling this again..
    clxq=y(w_ix)
    vq=y(w_ix+1)
    ayprime(w_ix)= vq
    ayprime(w_ix+1) = - 2*adotoa*vq - k*z*phidot - k**2*clxq - a**2*clxq*this%Vofphi(phi,2)

    end subroutine TQuintessence_PerturbationEvolve

    ! Early Quintessence example, axion potential from e.g. arXiv: 1908.06995

    function TEarlyQuintessence_has_switch(this, a) result(has_switch)
    class(TEarlyQuintessence) :: this
    logical has_switch
    real(dl) a
    
    if (this%use_fluid_approximation .and. a > this%a_fluid_switch) then
        has_switch = .true.
    else
        has_switch = .false.
    end if

    end function TEarlyQuintessence_has_switch

    subroutine TEarlyQuintessence_Switch(this, w_ix, a, k, z, y)
    class(TEarlyQuintessence), intent(inout) :: this
    real(dl), intent(in) :: a, k, z
    real(dl), intent(inout) :: y(:)    
    integer, intent(in) :: w_ix
    real(dl) grhov_fluid, gpres_fluid
    real(dl) :: mtilde, H, phic, phis, dphicdt, dphisdt, afluid, D, a2
    real(dl), parameter :: units = MPC_in_sec /Tpl  !convert to units of 1/Mpc
    real(dl) delphi, ddelphidt, dhsdt, delphic, delphis, ddelphicdt, ddelphisdt
    real(dl) dgrhoe, dgqe
    integer i

    if (this%n == 1 .and. this%use_PH) then
        a2 = a**2
        mtilde = units * this%m
        ! Derivatives with respect to cosmological time
        delphi=y(w_ix)
        ddelphidt=y(w_ix+1) / a
        dhsdt = 2 * k * z / a
        call this%calc_auxillary(a, grhov_fluid, gpres_fluid, phic, phis, dphisdt, dphicdt, D, H)
        delphic = delphi
        delphis = (2*delphi*(D + 3*H)*k**2 + a**2*(2*D**2*ddelphidt + 12*D*ddelphidt*H + 18*ddelphidt*H**2 + 4*(2*ddelphidt + 3*delphi*H)*mtilde**2 + 2*dhsdt*mtilde*(-dphisdt + mtilde*phic) + D*dhsdt*(dphicdt + mtilde*phis) + 3*dhsdt*H*(dphicdt + mtilde*phis)))/(2.*mtilde*(2*k**2 + a**2*(D**2 + 3*D*H + 4*mtilde**2))) 
        ddelphicdt = (2*(2*ddelphidt - delphi*(D + 3*H))*k**2 - a**2*(6*D*ddelphidt*H + 3*dhsdt*dphicdt*H + 6*H*(3*ddelphidt*H + 2*delphi*mtilde**2) + dhsdt*mtilde*(-2*dphisdt + 2*mtilde*phic + 3*H*phis) + D*dhsdt*(dphicdt + mtilde*phis)))/(4*k**2 + 2*a**2*(D**2 + 3*D*H + 4*mtilde**2)) 
        ddelphisdt = -0.5*(2*delphi*k**4 + a**2*k**2*(2*D*ddelphidt + 6*ddelphidt*H + 4*delphi*mtilde**2 + dhsdt*(dphicdt + mtilde*phis)) + a**4*mtilde*(12*ddelphidt*H*mtilde - 6*D*delphi*H*mtilde + D*dhsdt*(dphisdt - mtilde*phic) + 2*dhsdt*mtilde*(dphicdt + mtilde*phis)))/(a**2*mtilde*(2*k**2 + a**2*(D**2 + 3*D*H + 4*mtilde**2)))
        dgrhoe= 0.5d0 * a2 * (dphicdt*ddelphicdt + dphisdt*ddelphisdt + mtilde * (phis*ddelphicdt - phic*ddelphisdt) + mtilde * (delphis*dphicdt - delphic*dphisdt) + 2 * mtilde**2 * (phis*delphis  + phic*delphic))
        dgqe= 0.5d0 * k * a * (mtilde * (delphic*phis - delphis*phic) + delphic*dphicdt + delphis*dphisdt) 
        y(w_ix+2) = dgrhoe/grhov_fluid
        y(w_ix+3) = dgqe/grhov_fluid
    end if
    
    end subroutine TEarlyQuintessence_Switch


    subroutine calc_auxillary(this, a, grhov_fluid, gpres_fluid, phic, phis, dphisdt, dphicdt, D, H)
    class(TEarlyQuintessence), intent(inout) :: this
    real(dl), intent(in) :: a
    real(dl), intent(out) :: grhov_fluid, gpres_fluid, phic, phis, dphicdt, dphisdt, D, H
    real(dl) phi, phidot
    real(dl) :: mtilde, dHdt, dphidt, afluid, grho_tot, gpres_tot, a2
    real(dl), parameter :: units = MPC_in_sec /Tpl  !convert to units of 1/Mpc
    integer i
    call this%ValsAta(a, phi, phidot)
    a2 = a**2
    grhov_fluid = 0.5d0*phidot**2 + a2*this%Vofphi(phi,0,1)
    gpres_fluid = 0.5d0*phidot**2 - a2*this%Vofphi(phi,0,1)
    mtilde = units * this%m
    ! Derivatives with respect to cosmological time
    dphidt = phidot / a
    do i=1,3
        grho_tot = this%state%grho_no_de(a) + a2*grhov_fluid + a2*this%Vofphi(phi,0,2)
        gpres_tot = this%state%gpres_no_de(a) + a2*gpres_fluid - a2*this%Vofphi(phi,0,2)
        H = sqrt(grho_tot/3.0d0) / a2 
        dHdt =  - 0.5d0 * (grho_tot/3.0d0 + gpres_tot) / a**4 - H**2
        D = -H/2 * (3 - 2* dHdt / H**2)
        phic = phi
        phis = (dphidt*((D+3*H)**2+4*mtilde**2)+6*H*mtilde**2*phi)/(D*(D+3*H)*mtilde+4*mtilde**3)
        dphisdt = (3*H*mtilde*(-2*dphidt+D*phi))/(D**2+3*D*H+4*mtilde**2)
        dphicdt = (-3*H*(D*dphidt+3*dphidt*H+2*mtilde**2*phi))/(D**2+3*D*H+4*mtilde**2)
        grhov_fluid = 0.5d0 * a2*(0.5d0 * (dphisdt**2 + dphicdt**2) + mtilde * (-phic * dphisdt + phis*dphicdt) + mtilde**2 * (phic**2 + phis**2)) 
        gpres_fluid = 0.5d0 * a2*(0.5d0 * (dphisdt**2 + dphicdt**2) + mtilde * (-phic * dphisdt + phis*dphicdt)) 
    end do
    end subroutine calc_auxillary

    
    function TEarlyQuintessence_VofPhi(this, phi, deriv, component) result(VofPhi)
    !The input variable phi is sqrt(8*Pi*G)*psi
    !Returns (8*Pi*G)^(1-deriv/2)*d^{deriv}V(psi)/d^{deriv}psi evaluated at psi
    !return result is in 1/Mpc^2 units [so times (Mpc/c)^2 to get units in 1/Mpc^2]
    class(TEarlyQuintessence) :: this
    real(dl) phi,Vofphi
    integer deriv
    integer, intent(in), optional :: component !FULL_POTENTIAL = 0, QUINTESSENCE_ONLY = 1, LCDM_ONLY = 2
    real(dl) theta, costheta
    real(dl), parameter :: units = MPC_in_sec**2 /Tpl**2  !convert to units of 1/Mpc^2
    integer :: calc_component

    ! Determine which component(s) to calculate
    if (present(component)) then
        calc_component = component
    else
        calc_component = 0
    end if

    ! Initialize VofPhi
    VofPhi = 0.0_dl

    ! Calculate quintessence part if needed
    if (calc_component /= 2) then
        ! Assume f = sqrt(kappa)*f_theory = f_theory/M_pl
        ! m = m_theory/M_Pl
        theta = phi/this%f
        if (deriv==0) then
            Vofphi = VofPhi + units*this%m**2*this%f**2*(1 - cos(theta))**this%n + this%frac_lambda0*this%State%grhov
        else if (deriv ==1) then
            Vofphi = VofPhi + units*this%m**2*this%f*this%n*(1 - cos(theta))**(this%n-1)*sin(theta)
        else if (deriv ==2) then
            costheta = cos(theta)
            Vofphi = VofPhi + units*this%m**2*this%n*(1 - costheta)**(this%n-1)*(this%n*(1+costheta) -1)
        end if
    end if

     ! Add LCDM component if needed
    if (calc_component /= 1 .and. deriv == 0) then
        VofPhi = VofPhi + this%frac_lambda0*this%State%grhov
    end if

    end function TEarlyQuintessence_VofPhi

    subroutine TEarlyQuintessence_BackgroundDensityAndPressure(this, grhov, a, grhov_t, w)
    !Get grhov_t = 8*pi*rho_de*a**2 and (optionally) equation of state at scale factor a
    class(TEarlyQuintessence), intent(inout) :: this
    real(dl), intent(in) :: grhov, a
    real(dl), intent(out) :: grhov_t
    real(dl), optional, intent(out) :: w
    real(dl) V, a2, grhov_lambda, phi, phidot, grhov_fluid, w_fluid, dwdloga_fluid

    if (this%is_cosmological_constant) then
        grhov_t = grhov * a * a
        if (present(w)) w = -1_dl
    elseif (a >= this%astart) then

        a2 = a**2

        if (this%use_fluid_approximation .and. a > this%a_fluid_switch) then
            ! Use fluid approximation
            call this%FluidValsAta(a,grhov_fluid, w_fluid, dwdloga_fluid)
            V = this%Vofphi(0.0d0,0,2)
            grhov_t = grhov_fluid + a2*V
            if (present(w)) then
                ! This is the combined quintessence + lambda value
                w = (w_fluid * grhov_fluid - a2*V) / grhov_t
            end if
        else
            call this%ValsAta(a,phi,phidot)
            V = this%Vofphi(phi,0)
            grhov_t = phidot**2/2 + a2*V
            if (present(w)) then
                w = (phidot**2/2 - a2*V)/grhov_t
            end if
        end if
    else
        grhov_t=0
        if (present(w)) w = -1
    end if

    end subroutine TEarlyQuintessence_BackgroundDensityAndPressure

    subroutine EvolveBackgroundFluid(this,num,loga,y,yprime)
    class(TEarlyQuintessence) :: this
    integer num
    real(dl) y(num),yprime(num)
    real(dl) loga, a, a2, grhov_t, grhoa2, dtauda, w, H, mtilde
    real(dl), parameter :: units = MPC_in_sec /Tpl  !convert to units of 1/Mpc

    grhov_t = y(1)
    a = exp(loga)
    a2=a**2
    grhoa2 = this%state%grho_no_de(a) +  grhov_t * a2
    ! H in cosmological time 
    H = sqrt(grhoa2/3.0d0) /a2
    mtilde = units * this%m 
    if (this%n == 1 .and. this%use_PH) then
        w = 1.5d0 * (mtilde/H)**(-2)
    else
        w = (this%n - 1.0d0) / (this%n + 1.0d0)
    end if
    yprime(1)= 2*grhov_t/a - 3*(1+w)/a*grhov_t 
    yprime(1) = yprime(1)*a

    end subroutine EvolveBackgroundFluid


    subroutine FluidValsAta(this,a,agrhov_fluid, aw_fluid, adwdloga_fluid)
    class(TEarlyQuintessence) :: this
    !Do interpolation for background phi and phidot at a (precomputed in Init)
    real(dl) a, agrhov_fluid, aw_fluid, adwdloga_fluid
    real(dl) a0,b0,ho2o6,delta,da
    integer ix

    if (a >= 0.999d0) then
        agrhov_fluid = this%grhov_fluid(this%npoints_fluid)
        aw_fluid = this%w_fluid(this%npoints_fluid)
        adwdloga_fluid = this%dwdloga_fluid(this%npoints_fluid)
        return
    endif

    delta= log(a)-log(this%a_fluid_switch)
    ix = int(delta/this%dloga_fluid)+1
    da = this%sampled_a_fluid(ix+1) - this%sampled_a_fluid(ix)
    a0 = (this%sampled_a_fluid(ix+1) - a)/da
    b0 = 1 - a0
    ho2o6 = da**2/6._dl
    agrhov_fluid=b0*this%grhov_fluid(ix+1) + a0*(this%grhov_fluid(ix)-b0*((a0+1)*this%ddgrhov_fluid(ix)+(2-a0)*this%ddgrhov_fluid(ix+1))*ho2o6)
    aw_fluid=b0*this%w_fluid(ix+1) + a0*(this%w_fluid(ix)-b0*((a0+1)*this%ddw_fluid(ix)+(2-a0)*this%ddw_fluid(ix+1))*ho2o6)
    adwdloga_fluid=b0*this%dwdloga_fluid(ix+1) + a0*(this%dwdloga_fluid(ix)-b0*((a0+1)*this%dddwdloga_fluid(ix)+(2-a0)*this%dddwdloga_fluid(ix+1))*ho2o6)

    end subroutine FluidValsAta

    subroutine TEarlyQuintessence_PerturbedStressEnergy(this, dgrhoe, dgqe, &
        a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1, ay, ayprime, w_ix)
    !Get density perturbation and heat flux
    class(TEarlyQuintessence), intent(inout) :: this
    real(dl), intent(out) :: dgrhoe, dgqe
    real(dl), intent(in) ::  a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1
    real(dl), intent(in) :: ay(*)
    real(dl), intent(inout) :: ayprime(*)
    integer, intent(in) :: w_ix
    real(dl) phi, phidot, delta_phi, delta_phidot, grhov_fluid, w_fluid, dwdloga_fluid

    delta_phi=ay(w_ix)
    delta_phidot=ay(w_ix+1)

    if (this%use_fluid_approximation .and. a > this%a_fluid_switch) then

        call this%FluidValsAta(a, grhov_fluid, w_fluid, dwdloga_fluid)

        dgrhoe = ay(w_ix+2)*grhov_fluid
        dgqe= ay(w_ix+3)*grhov_fluid

    else

        call this%ValsAta(a,phi,phidot)
        dgrhoe= phidot*delta_phidot + delta_phi*a**2*this%Vofphi(phi,1)
        dgqe= k*phidot*delta_phi

    endif

    end subroutine TEarlyQuintessence_PerturbedStressEnergy

    subroutine TEarlyQuintessence_PerturbationEvolve(this, ayprime, w, w_ix, &
        a, adotoa, k, z, y)
    !Get conformal time derivatives of the density perturbation and velocity
    class(TEarlyQuintessence), intent(in) :: this
    real(dl), intent(inout) :: ayprime(:)
    real(dl), intent(in) :: a, adotoa, w, k, z, y(:)
    integer, intent(in) :: w_ix
    real(dl) clxq, uq, phi, phidot, cs2_fluid, Hv3_over_k, delta_phi, delta_phidot, phidotdot, grhov_t, V, Vprime, Vprimeprime, a2, grhov_fluid, w_fluid, dwdloga_fluid, mtilde, deriv, alpha 
    real(dl), parameter :: units = MPC_in_sec /Tpl  !convert to units of 1/Mpc s

    delta_phi=y(w_ix)
    delta_phidot=y(w_ix+1)

    call this%ValsAta(a,phi,phidot) !wasting time calling this again..

    if (this%use_fluid_approximation .and. a > this%a_fluid_switch) then
        ! Set scalar field derivatives to zero
        ayprime(w_ix)= 0
        ayprime(w_ix+1) = 0
    else
        ayprime(w_ix)= delta_phidot
        ayprime(w_ix+1) = - 2*adotoa*delta_phidot - k*z*phidot - k**2*delta_phi - a**2*delta_phi*this%Vofphi(phi,2)
    end if

    a2 = a**2
    
    clxq=y(w_ix+2)
    ! Heat flux u = (1+w)v
    uq=y(w_ix+3)

    if (this%use_fluid_approximation .and. a > this%a_fluid_switch) then

        call this%FluidValsAta(a, grhov_fluid, w_fluid, dwdloga_fluid)

        ! (19, 20) in https://arxiv.org/pdf/1410.2896 but generalised to cs2 /= 1
        ! ca2 = cs2 - 1/3 * deriv, where deriv =  dw/dlog a/(1+w)
        deriv = dwdloga_fluid / (1 + w_fluid)
        mtilde = units * this%m
        if (this%n == 1) then
            if (this%use_PH) then
                alpha = (k / (a * mtilde))**2
                cs2_fluid =  ((1 + alpha)**0.5 - 1)**2 / alpha + 5.0/4.0 * (adotoa / (a * mtilde))**2
            else
                cs2_fluid = k**2 / (k**2 + 4*mtilde**2*a2)
            end if
        else
            stop 'Not implemented'
        end if

        Hv3_over_k =  3*adotoa*uq / k

        !density perturbation
        ayprime(w_ix+2) = -3 * adotoa * (cs2_fluid - w_fluid) *  (y(w_ix+2) + Hv3_over_k) &
            -   k * y(w_ix + 3) - (1 + w_fluid) * k * z  - adotoa*deriv* Hv3_over_k
        !(1+w)v
        ayprime(w_ix + 3) = -adotoa * (1 - 3 * cs2_fluid -  deriv) * y(w_ix + 3) + &
            k * cs2_fluid * y(w_ix+2)

    else

        ! Quintessence component
        V = this%Vofphi(phi,0, 1) 
        Vprime = this%Vofphi(phi,1) 
        Vprimeprime = this%Vofphi(phi,2) 
        phidotdot = - 2 * adotoa * phidot - a2*Vprime
        grhov_t = phidot**2/2 + a2*V

        ! density and velocity perturbation in terms of field
        ayprime(w_ix+2) = (phidotdot * delta_phidot + phidot * ayprime(w_ix+1) + delta_phidot * a2 * Vprime + & 
        2 * delta_phi * adotoa * a2 * Vprime + delta_phi * a2 * Vprimeprime * phidot) / grhov_t - & 
        (phidot * phidotdot + 2 * adotoa * a2 * V + a2 * Vprime * phidot) * (phidot * delta_phidot + delta_phi * a2 * Vprime) / grhov_t**2
        ayprime(w_ix+3) = k * (phidotdot * delta_phi + phidot * delta_phidot) / grhov_t - & 
        (phidot * phidotdot + 2 * adotoa * a2 * V + a2 * Vprime * phidot) * k * phidot * delta_phi / grhov_t**2 

    endif


    end subroutine TEarlyQuintessence_PerturbationEvolve

    subroutine TEarlyQuintessence_Init(this, State)
    use Powell
    class(TEarlyQuintessence), intent(inout) :: this
    class(TCAMBdata), intent(in), target :: State
    real(dl) aend, afrom
    integer, parameter ::  NumEqs=2
    real(dl) c(24),w(NumEqs,9), y(NumEqs)
    integer, parameter ::  NumEqsFluid=1
    real(dl) cf(24),wf(NumEqsFluid,9), yf(NumEqsFluid)
    integer ind, i, ix
    real(dl), parameter :: splZero = 0._dl
    real(dl) lastsign, da_osc, last_a, a_c
    integer :: oscillation_count
    logical :: threshold_reached
    real(dl) initial_phi, initial_phidot, a, a2
    real(dl), dimension(:), allocatable :: sampled_a, phi_a, phidot_a, fde
    integer npoints, tot_points, max_ix
    logical has_peak
    real(dl) fzero, xzero
    integer iflag, iter
    Type(TTimer) :: Timer
    Type(TNEWUOA) :: Minimize
    real(dl) log_params(2), param_min(2), param_max(2)
    real(dl) phi_switch, phidot_switch, grhov_fluid_switch, gpres_fluid_switch
    real(dl) :: mtilde, H, dHdt, grho_tot, gpres_tot, grhoa2
    real(dl) :: phic_switch, phis_switch, dphisdt_switch, dphicdt_switch, D_switch, H_switch
    real(dl), parameter :: units = MPC_in_sec /Tpl  !convert to units of 1/Mpc

    !Make interpolation table, etc,
    !At this point massive neutrinos have been initialized
    !so grho_no_de can be used to get density and pressure of other components at scale factor a

    call this%TQuintessence%Init(State)

    this%a_fluid_switch = 1._dl
    threshold_reached = .false.
    ! Evolve both scalar field and fluid equations
    this%num_perturb_equations = 4

    if (this%use_zc) then
        !Find underlying parameters m,f to give specified zc and fde_zc (peak early dark energy fraction)
        !Input m,f are used as starting values for search, which is done by brute force
        !(so should generalize easily, but not optimized for this specific potential)
        log_params(1) = log(this%f)
        log_params(2) = log(this%m)

        if (.false.) then
            ! Can just iterate linear optimizations when nearly orthogonal
            call Timer%Start()
            do iter = 1, 2
                call brentq(this,match_fde,log(0.01_dl),log(10._dl), 1d-3,xzero,fzero,iflag)
                if (iflag/=0) print *, 'BRENTQ FAILED f'
                this%f = exp(xzero)
                print *, 'match to m, f =', this%m, this%f, fzero
                call brentq(this,match_zc,log(1d-55),log(1d-52), 1d-3,xzero,fzero,iflag)
                if (iflag/=0) print *, 'BRENTQ FAILED m'
                this%m = exp(xzero)
                print *, 'match to m, f =', this%m, this%f, fzero
                call this%calc_zc_fde(fzero, xzero)
                print *, 'matched outputs', fzero, xzero
            end do
            call Timer%WriteTime('Timing for fitting')
        end if
        if (this%DebugLevel>0) call Timer%Start()
        !Minimize in log f, log m
        ! param_min(1) = log(0.001_dl)
        ! param_min(2) = log(1d-58)
        ! param_max(1) = log(1e5_dl)
        ! param_max(2) = log(1d-50)
        ! if (Minimize%BOBYQA(this, match_fde_zc, 2, 5, log_params,param_min, &
        !           param_max, 0.8_dl,1e-4_dl,this%DebugLevel,2000)) then

        if (Minimize%NEWUOA(this, match_fde_zc, 2, 5, log_params,&
            0.8_dl,1e-4_dl,this%DebugLevel,500)) then

            if (Minimize%Last_bestfit > 1e-3) then
                global_error_flag = error_darkenergy
                global_error_message= 'TEarlyQuintessence ERROR converging solution for fde, zc'
                write(*,*) 'last-bestfit= ', Minimize%Last_bestfit
                return
            end if
            this%f = exp(log_params(1))
            this%m = exp(log_params(2))
            if (this%DebugLevel>0) then
                call this%calc_zc_fde(fzero, xzero)
                write(*,*) 'matched outputs Bobyqa zc, fde = ', fzero, xzero
            end if
        else
            global_error_flag = error_darkenergy
            global_error_message= 'TEarlyQuintessence ERROR finding solution for fde, zc'
            return
        end if
        if (this%DebugLevel>0) call Timer%WriteTime('Timing for parameter fitting')
    end if

    this%dloga = (-this%log_astart)/(this%npoints-1)

    !use log spacing in a up to max_a_log, then linear. Switch where step matches
    this%max_a_log = 1.d0/this%npoints/(exp(this%dloga)-1)
    npoints = (log(this%max_a_log)-this%log_astart)/this%dloga + 1

    if (allocated(this%phi_a)) then
        deallocate(this%phi_a,this%phidot_a)
        deallocate(this%ddphi_a,this%ddphidot_a, this%sampled_a)
    end if
    allocate(phi_a(npoints),phidot_a(npoints), sampled_a(npoints), fde(npoints))

    !initial_phi  = 10  !  0.3*grhom/m**3
    !initial_phi2 = 100 !   6*grhom/m**3
    !
    !!           initial_phi  = 65 !  0.3*grhom/m**3
    !!           initial_phi2 = 65 !   6*grhom/m**3
    !
    !astart=1d-9
    !
    !!See if initial conditions are giving correct omega_de now
    !atol=1d-8
    !initial_phidot =  astart*this%phidot_start(this%initial_phi)
    !om1= this%GetOmegaFromInitial(astart,initial_phi,initial_phidot, atol)
    !
    !print*, State%omega_de, 'first trial:', om1
    !if (abs(om1-State%omega_de > this%omega_tol)) then
    !    !if not, do binary search in the interval
    !    OK=.false.
    !    initial_phidot = astart*this%phidot_start(initial_phi2)
    !    om2= this%GetOmegaFromInitial(astart,initial_phi2,initial_phidot, atol)
    !    if (om1 > State%omega_de .or. om2 < State%omega_de) then
    !        write (*,*) 'initial phi values must bracket required value.  '
    !        write (*,*) 'om1, om2 = ', real(om1), real(om2)
    !        stop
    !    end if
    !    do iter=1,100
    !        deltaphi=initial_phi2-initial_phi
    !        phi =initial_phi + deltaphi/2
    !        initial_phidot =  astart*Quint_phidot_start(phi)
    !        om = this%GetOmegaFromInitial(astart,phi,initial_phidot,atol)
    !        if (om < State%omega_de) then
    !            om1=om
    !            initial_phi=phi
    !        else
    !            om2=om
    !            initial_phi2=phi
    !        end if
    !        if (om2-om1 < 1d-3) then
    !            OK=.true.
    !            initial_phi = (initial_phi2+initial_phi)/2
    !            if (FeedbackLevel > 0) write(*,*) 'phi_initial = ',initial_phi
    !            exit
    !        end if
    !
    !    end do !iterations
    !    if (.not. OK) stop 'Search for good intial conditions did not converge' !this shouldn't happen
    !
    !end if !Find initial

    initial_phi = this%theta_i*this%f

    y(1)=initial_phi
    initial_phidot =  this%astart*this%phidot_start(initial_phi)
    y(2)= initial_phidot*this%astart**2

    phi_a(1)=y(1)
    phidot_a(1)=y(2)/this%astart**2
    sampled_a(1)=this%astart
    da_osc = 1
    last_a = this%astart
    max_ix =0

    ind=1
    afrom=this%log_astart
    do i=1, npoints-1
        aend = this%log_astart + this%dloga*i
        ix = i+1
        sampled_a(ix)=exp(aend)
        a2 = sampled_a(ix)**2
        call dverk(this,NumEqs,EvolveBackgroundLog,afrom,y,aend,this%integrate_tol,ind,c,NumEqs,w)
        if (.not. this%check_error(exp(afrom), exp(aend))) return
        call EvolveBackgroundLog(this,NumEqs,aend,y,w(:,1))
        phi_a(ix)=y(1)
        phidot_a(ix)=y(2)/a2
        !if (i==1) then
        !    lastsign = y(2)
        !elseif (y(2)*lastsign < 0) then
        !    !derivative has changed sign. Use to probe any oscillation scale:
        !    da_osc = min(da_osc, exp(aend) - last_a)
        !    last_a = exp(aend)
        !    lastsign= y(2)
        !end if

        ! Check for sign change in phidot (half-cycle of oscillation)
        if (i==1) then
            lastsign = sign(1.0_dl, phidot_a(ix))
            oscillation_count = 0
        elseif (sign(1.0_dl, phidot_a(ix)) /= lastsign) then
            oscillation_count = oscillation_count + 1
            lastsign = sign(1.0_dl, phidot_a(ix))
            ! Update da_osc
            da_osc = min(da_osc, exp(aend) - last_a)
            last_a = exp(aend)
            if (.not. threshold_reached .and. oscillation_count / 2 >= this%oscillation_threshold) then
                this%a_fluid_switch = sampled_a(ix)
                threshold_reached = .true.
                if (this%DebugLevel > 0) then
                    write(*,*) 'TEarlyQuintessence: Switching to fluid approximation at a =', this%a_fluid_switch
                end if
            end if
        end if

        !Define fde as ratio of early dark energy density to total
        fde(ix) = 1/((this%state%grho_no_de(sampled_a(ix)) +  this%frac_lambda0*this%State%grhov*a2**2) &
            /(a2*(0.5d0* phidot_a(ix)**2 + a2*this%Vofphi(y(1),0))) + 1)
        if (max_ix==0 .and. ix > 2 .and. fde(ix)< fde(ix-1)) then
            max_ix = ix-1
        end if
        if (sampled_a(ix)*(exp(this%dloga)-1)*this%min_steps_per_osc > da_osc) then
            !Step size getting too big to sample oscillations well
            exit
        end if
    end do

    if (this%DebugLevel > 0) then
        write(*,*) 'TEarlyQuintessence: Number of oscillation cycles at linear switch:', oscillation_count/2
    end if

    ! Do remaining steps with linear spacing in a, trying to be small enough
    this%npoints_log = ix
    this%max_a_log = sampled_a(ix)
    this%da = min(this%max_a_log *(exp(this%dloga)-1), &
        da_osc/this%min_steps_per_osc, (1- this%max_a_log)/(this%npoints-this%npoints_log))
    this%da = max(this%da, 1e-7)
    this%npoints_linear = int((1- this%max_a_log)/ this%da)+1
    this%da = (1- this%max_a_log)/this%npoints_linear

    tot_points = this%npoints_log+this%npoints_linear
    allocate(this%phi_a(tot_points),this%phidot_a(tot_points))
    allocate(this%ddphi_a(tot_points),this%ddphidot_a(tot_points))
    allocate(this%sampled_a(tot_points), this%fde(tot_points), this%ddfde(tot_points))
    this%sampled_a(1:ix) = sampled_a(1:ix)
    this%phi_a(1:ix) = phi_a(1:ix)
    this%phidot_a(1:ix) = phidot_a(1:ix)
    this%sampled_a(1:ix) = sampled_a(1:ix)
    this%fde(1:ix) = fde(1:ix)

    ind=1
    afrom = this%max_a_log
    do i=1, this%npoints_linear
        ix = this%npoints_log + i
        aend = this%max_a_log + this%da*i
        a2 =aend**2
        this%sampled_a(ix)=aend

        if (this%use_fluid_approximation .and. aend > this%a_fluid_switch) then

            this%phi_a(ix) = 0.0_dl
            this%phidot_a(ix)= 0.0_dl

        else

            call dverk(this,NumEqs,EvolveBackground,afrom,y,aend,this%integrate_tol,ind,c,NumEqs,w)
            if (.not. this%check_error(afrom, aend)) return
            call EvolveBackground(this,NumEqs,aend,y,w(:,1))
            this%phi_a(ix)=y(1)
            this%phidot_a(ix)=y(2)/a2
            ! Check for sign change in phidot (half-cycle of oscillation)
            if (sign(1.0_dl, this%phidot_a(ix)) /= lastsign) then
                oscillation_count = oscillation_count + 1
                lastsign = sign(1.0_dl, this%phidot_a(ix))
                if (.not. threshold_reached .and. oscillation_count / 2 >= this%oscillation_threshold) then
                    this%a_fluid_switch = this%sampled_a(ix)
                    threshold_reached = .true.
                    if (this%DebugLevel > 0) then
                        write(*,*) 'Switching to fluid approximation at a =', this%a_fluid_switch
                    end if
                end if
            end if

        endif

        this%fde(ix) = 1/((this%state%grho_no_de(aend) +  this%frac_lambda0*this%State%grhov*a2**2) &
            /(a2*(0.5d0* this%phidot_a(ix)**2 + a2*this%Vofphi(this%phi_a(ix),0))) + 1)

        if (max_ix==0 .and. this%fde(ix)< this%fde(ix-1)) then
            max_ix = ix-1
        end if
    end do

    ! Print the result
    if (this%DebugLevel > 0) then
        write(*,*) 'TEarlyQuintessence: Number of oscillation cycles:', oscillation_count/2
    end if

    call spline(this%sampled_a,this%phi_a,tot_points,splZero,splZero,this%ddphi_a)
    call spline(this%sampled_a,this%phidot_a,tot_points,splZero,splZero,this%ddphidot_a)
    call spline(this%sampled_a,this%fde,tot_points,splZero,splZero,this%ddfde)
    has_peak = .false.
    if (max_ix >0) then
        ix = max_ix
        has_peak = this%fde_peak(a_c, this%sampled_a(ix), this%sampled_a(ix+1), this%fde(ix), &
            this%fde(ix+1), this%ddfde(ix), this%ddfde(ix+1))
        if (.not. has_peak) then
            has_peak = this%fde_peak(a_c, this%sampled_a(ix-1), this%sampled_a(ix), &
                this%fde(ix-1), this%fde(ix), this%ddfde(ix-1), this%ddfde(ix))
        end if
    end if
    if (has_peak) then
        this%zc = 1/a_c-1
        this%fde_zc = this%fdeAta(a_c)
    else
        if (this%DebugLevel>0) write(*,*) 'TEarlyQuintessence: NO PEAK '
        this%zc = -1
    end if
    if (this%DebugLevel>0) then
        write(*,*) 'TEarlyQuintessence zc, fde used', this%zc, this%fde_zc
    end if

    ! Passaglia and Hu 

    ! Testing
    !if (this%oscillation_threshold < 10) then
    !    this%a_fluid_switch = 5.0e-05
    !end if

    if (this%n == 1 .and. this%use_PH) then
        call this%calc_auxillary(this%a_fluid_switch, grhov_fluid_switch, gpres_fluid_switch, phic_switch, phis_switch, dphisdt_switch, dphicdt_switch, D_switch, H_switch)
    else
        call this%ValsAta(this%a_fluid_switch, phi_switch, phidot_switch)
        grhov_fluid_switch = 0.5d0*phidot_switch**2 + this%a_fluid_switch**2*this%Vofphi(phi_switch,0,1)
        gpres_fluid_switch = 0.5d0*phidot_switch**2 - this%a_fluid_switch**2*this%Vofphi(phi_switch,0,1)
    end if

    this%npoints_fluid = 100000

    if (allocated(this%sampled_a_fluid)) then
        deallocate(this%sampled_a_fluid, this%grhov_fluid, this%ddgrhov_fluid, this%w_fluid, this%ddw_fluid, this%dwdloga_fluid, this%dddwdloga_fluid)
    end if

    allocate(this%sampled_a_fluid(this%npoints_fluid), this%grhov_fluid(this%npoints_fluid), this%ddgrhov_fluid(this%npoints_fluid))
    allocate(this%w_fluid(this%npoints_fluid), this%ddw_fluid(this%npoints_fluid), this%dwdloga_fluid(this%npoints_fluid), this%dddwdloga_fluid(this%npoints_fluid))

    this%dloga_fluid = -log(this%a_fluid_switch)/this%npoints_fluid

    mtilde = units * this%m

    yf(1) = grhov_fluid_switch

    ind=1
    afrom = log(this%a_fluid_switch)
    do i=1, this%npoints_fluid
        aend = log(this%a_fluid_switch) +  this%dloga_fluid*i
        this%sampled_a_fluid(i) = exp(aend)
        call dverk(this,NumEqsFluid,EvolveBackgroundFluid,afrom,yf,aend,this%integrate_tol,ind,cf,NumEqsFluid,wf)
        if (.not. this%check_error(afrom, aend)) return
        call EvolveBackgroundFluid(this,NumEqsFluid,aend,yf,wf(:,1))
        this%grhov_fluid(i) = yf(1)
        ! Repeated here - couldn't return w from function
        a = exp(aend)
        a2=a**2
        grho_tot = this%state%grho_no_de(a) +  yf(1) * a2 + a2*this%Vofphi(0.0d0,0,2) ! CC so use any phi
        ! H in cosmological time 
        H = sqrt(grho_tot/3.0d0) /a2
        dHdt =  - 0.5d0 * (grho_tot/3.0d0 + gpres_tot) / this%a_fluid_switch**4 - H**2
        if (this%n == 1 .and. this%use_PH) then
            this%w_fluid(i)  = 1.5d0 * (mtilde/H)**(-2)
        else
            this%w_fluid(i) = (this%n - 1.0d0) / (this%n + 1.0d0)
        end if
        gpres_tot = this%state%gpres_no_de(a) + this%w_fluid(i) * yf(1) * a2 - a2*this%Vofphi(0.0d0,0,2)
        dHdt =  - 0.5d0 * (grho_tot/3.0d0 + gpres_tot) / a**4 - H**2
        if (this%n == 1 .and. this%use_PH) then
            this%dwdloga_fluid(i) = 3.0d0 * dHdt / mtilde**2
        else
            this%dwdloga_fluid(i) = 0.0d0
        end if
        !write(*,*) this%sampled_a_fluid(i), this%grhov_fluid(i), this%w_fluid(i), this%dwdloga_fluid(i)  
    end do

    call spline(this%sampled_a_fluid,this%grhov_fluid,this%npoints_fluid,splZero,splZero,this%ddgrhov_fluid)
    call spline(this%sampled_a_fluid,this%w_fluid,this%npoints_fluid,splZero,splZero,this%ddw_fluid)

    if (this%DebugLevel>0) then
        write(*,*) 'TEarlyQuintessence finished init'
    end if

    end subroutine TEarlyQuintessence_Init

    logical function check_error(this, afrom, aend)
    class(TEarlyQuintessence) :: this
    real(dl) afrom, aend

    if (global_error_flag/=0) then
        write(*,*) 'TEarlyQuintessence error integrating', afrom, aend
        write(*,*) this%n, this%f, this%m, this%theta_i
        stop
        check_error = .false.
        return
    end if
    check_error= .true.
    end function check_error

    logical function fde_peak(this, peak, xlo, xhi, Flo, Fhi, ddFlo, ddFhi)
    class(TEarlyQuintessence) :: this
    real(dl), intent(out) :: peak
    real(dl) Delta
    real(dl), intent(in) :: xlo, xhi, ddFlo, ddFhi,Flo, Fhi
    real(dl) a, b, c, fac

    !See if derivative has zero in spline interval xlo .. xhi

    Delta = xhi - xlo

    a = 0.5_dl*(ddFhi-ddFlo)/Delta
    b = (xhi*ddFlo-xlo*ddFhi)/Delta
    c = (Fhi-Flo)/Delta+ Delta/6._dl*((1-3*xhi**2/Delta**2)*ddFlo+(3*xlo**2/Delta**2-1)*ddFhi)
    fac = b**2-4*a*c
    if (fac>=0) then
        fac = sqrt(fac)
        peak = (-b + fac)/2/a
        if (peak >= xlo .and. peak <= xhi) then
            fde_peak = .true.
            return
        else
            peak = (-b - fac)/2/a
            if (peak >= xlo .and. peak <= xhi) then
                fde_peak = .true.
                return
            end if
        end if
    end if
    fde_peak = .false.

    end function fde_peak

    function match_zc(this, logm)
    class(TEarlyQuintessence), intent(inout) :: this
    real(dl), intent(in) :: logm
    real(dl) match_zc, zc, fde_zc

    this%m = exp(logm)
    call this%calc_zc_fde(zc, fde_zc)
    match_zc = zc - this%zc

    end function match_zc

    function match_fde(this, logf)
    class(TEarlyQuintessence), intent(inout) :: this
    real(dl), intent(in) :: logf
    real(dl) match_fde, zc, fde_zc

    this%f = exp(logf)
    call this%calc_zc_fde(zc, fde_zc)
    match_fde = fde_zc - this%fde_zc

    end function match_fde

    function match_fde_zc(this, x)
    class(TEarlyQuintessence) :: this
    real(dl), intent(in) :: x(:)
    real(dl) match_fde_zc, zc, fde_zc

    this%f = exp(x(1))
    this%m = exp(x(2))
    call this%calc_zc_fde(zc, fde_zc)

    match_fde_zc = (log(this%fde_zc)-log(fde_zc))**2 + (log(zc)-log(this%zc))**2
    if (this%DebugLevel>1) then
        write(*,*) 'search f, m, zc, fde_zc, chi2', this%f, this%m, zc, fde_zc, match_fde_zc
    end if

    end function match_fde_zc

    subroutine calc_zc_fde(this, z_c, fde_zc)
    class(TEarlyQuintessence), intent(inout) :: this
    real(dl), intent(out) :: z_c, fde_zc
    real(dl) aend, afrom
    integer, parameter ::  NumEqs=2
    real(dl) c(24),w(NumEqs,9), y(NumEqs)
    integer ind, i, ix
    real(dl), parameter :: splZero = 0._dl
    real(dl) a_c
    real(dl) initial_phi, initial_phidot, a2
    real(dl), dimension(:), allocatable :: sampled_a, fde, ddfde
    integer npoints, max_ix
    logical has_peak
    real(dl) a0, b0, da

    ! Get z_c and f_de(z_c) where z_c is the redshift of (first) peak of f_de (de energy fraction)
    ! Do this by forward propagating until peak, then get peak values by cubic interpolation

    initial_phi = this%theta_i*this%f
    this%log_astart = log(this%astart)
    this%dloga = (-this%log_astart)/(this%npoints-1)

    npoints = (-this%log_astart)/this%dloga + 1
    allocate(sampled_a(npoints), fde(npoints), ddfde(npoints))

    y(1)=initial_phi
    initial_phidot =  this%astart*this%phidot_start(initial_phi)
    y(2)= initial_phidot*this%astart**2
    sampled_a(1)=this%astart
    max_ix =0
    ind=1
    afrom=this%log_astart
    do i=1, npoints-1
        aend = this%log_astart + this%dloga*i
        ix = i+1
        sampled_a(ix)=exp(aend)
        a2 = sampled_a(ix)**2
        call dverk(this,NumEqs,EvolveBackgroundLog,afrom,y,aend,this%integrate_tol,ind,c,NumEqs,w)
        if (.not. this%check_error(exp(afrom), exp(aend))) return
        call EvolveBackgroundLog(this,NumEqs,aend,y,w(:,1))
        fde(ix) = 1/((this%state%grho_no_de(sampled_a(ix)) +  this%frac_lambda0*this%State%grhov*a2**2) &
            /((0.5d0*y(2)**2/a2 + a2**2*this%Vofphi(y(1),0))) + 1)
        if (max_ix==0 .and. ix > 2 .and. fde(ix)< fde(ix-1)) then
            max_ix = ix-1
        end if
        if (max_ix/=0 .and. ix > max_ix+4) exit
    end do

    call spline(sampled_a,fde,ix,splZero,splZero,ddfde)
    has_peak = .false.
    if (max_ix >0) then
        has_peak = this%fde_peak(a_c, sampled_a(max_ix), sampled_a(max_ix+1), fde(max_ix), &
            fde(max_ix+1), ddfde(max_ix), ddfde(max_ix+1))
        if (.not. has_peak) then
            has_peak = this%fde_peak(a_c, sampled_a(max_ix-1), sampled_a(max_ix), &
                fde(max_ix-1), fde(max_ix), ddfde(max_ix-1), ddfde(max_ix))
        end if
    end if
    if (has_peak) then
        z_c = 1/a_c-1
        ix = int((log(a_c)-this%log_astart)/this%dloga)+1
        da = sampled_a(ix+1) - sampled_a(ix)
        a0 = (sampled_a(ix+1) - a_c)/da
        b0 = 1 - a0
        fde_zc=b0*fde(ix+1) + a0*(fde(ix)-b0*((a0+1)*ddfde(ix)+(2-a0)*ddfde(ix+1))*da**2/6._dl)
    else
        write(*,*) 'calc_zc_fde: NO PEAK'
        z_c = -1
        fde_zc = 0
    end if

    end subroutine calc_zc_fde

    function fdeAta(this,a)
    class(TEarlyQuintessence) :: this
    real(dl), intent(in) :: a
    real(dl) fdeAta, aphi, aphidot, a2

    call this%ValsAta(a, aphi, aphidot)
    a2 = a**2
    fdeAta = 1/((this%state%grho_no_de(a) +  this%frac_lambda0*this%State%grhov*a2**2) &
        /(a2*(0.5d0* aphidot**2 + a2*this%Vofphi(aphi,0))) + 1)
    end function fdeAta

    subroutine TEarlyQuintessence_ReadParams(this, Ini)
    use IniObjects
    class(TEarlyQuintessence) :: this
    class(TIniFile), intent(in) :: Ini

    call this%TDarkEnergyModel%ReadParams(Ini)

    end subroutine TEarlyQuintessence_ReadParams


    function TEarlyQuintessence_PythonClass()
    character(LEN=:), allocatable :: TEarlyQuintessence_PythonClass

    TEarlyQuintessence_PythonClass = 'EarlyQuintessence'

    end function TEarlyQuintessence_PythonClass

    subroutine TEarlyQuintessence_SelfPointer(cptr,P)
    use iso_c_binding
    Type(c_ptr) :: cptr
    Type (TEarlyQuintessence), pointer :: PType
    class (TPythonInterfacedClass), pointer :: P

    call c_f_pointer(cptr, PType)
    P => PType

    end subroutine TEarlyQuintessence_SelfPointer


    !real(dl) function GetOmegaFromInitial(this, astart,phi,phidot,atol)
    !!Get omega_de today given particular conditions phi and phidot at a=astart
    !class(TQuintessence) :: this
    !real(dl), intent(IN) :: astart, phi,phidot, atol
    !integer, parameter ::  NumEqs=2
    !real(dl) c(24),w(NumEqs,9), y(NumEqs), ast
    !integer ind, i
    !
    !ast=astart
    !ind=1
    !y(1)=phi
    !y(2)=phidot*astart**2
    !call dverk(this,NumEqs,EvolveBackground,ast,y,1._dl,atol,ind,c,NumEqs,w)
    !call EvolveBackground(this,NumEqs,1._dl,y,w(:,1))
    !
    !GetOmegaFromInitial=(0.5d0*y(2)**2 + Vofphi(y(1),0))/this%State%grhocrit !(3*adot**2)
    !
    !end function GetOmegaFromInitial
    end module Quintessence
