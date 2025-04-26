#
# update a phoenix/1D dqs file to produce RFs
# uses a template
# this version is for use with an slurm array
# job
#
import os
import sys
import re
import f90nml
import fileinput
import glob
import math

def calcMixingLength(Teff, logg):
  # constants
  a0 = 1.587
  a1 = -0.05
  a2 = 0.045
  a3 = -0.039
  a4 = 0.176
  a5 = -0.067
  a6 = 0.107
  # scale Teff and logg
  Ts = (Teff - 5770.) / 1000.
  gs = math.log10(10**logg / 27500.)
  # calculate
  alpha = a0 + (a1 + (a3 + a5*Ts + a6*gs)*Ts + a4*gs)*Ts + a2*gs
  # check boundaries
  if alpha < 1.: alpha = 1.
  if alpha > 3.5: alpha = 3.5
  # return it
  return alpha


def calcMass(Teff, logg):
  if logg > 4:      Msun = 1
  elif logg > 3:    Msun = 1.2
  elif logg > 2:    Msun = 1.4
  elif logg > 1.6:  Msun = 2.
  elif logg > 0.9:  Msun = 3.
  elif logg > 0:    Msun = 4.
  else:             Msun = 5.
  return Msun * (Teff/5770.)**2



def update_dqs(filename,template,file_stem,n_MPI, Teff_new=-1., logg_new=-999.):
        """
          update dqs files from a restart file and
          a template while changing the template to
          reflect important changes (Teff, R0, v0)

          filename: name of restart (.20) file to use
          template: name of the template dqs file (note: use "" to mask env. vars, e.g., 'rm -r "$TMPDIR/$JOB_NAME"')
          file_stem: stemp of the created dqs file, .e.g., '.aurora.dqs'
          n_MPI: number of MPI procs to use (tasks-per-node)
        """
#
# read the namelist from the .20 file to
# get its paramters (Teff, ...)
#
        target_nml = f90nml.read(filename)
        nwrk = 1
        nwl =  -1

#    for i in range(0,6):
#        eheu_Be = 1.38 + i
#
# extract relevant values:
# add more lines as needed!
#

        teff = target_nml['phoenix']['teff']
        r0 = target_nml['phoenix']['r0']
        v0 = target_nml['phoenix']['v0']
        logg = target_nml['phoenix']['logg']
        zscale = target_nml['phoenix']['zscale']
        alpha_scale = target_nml['phoenix']['alpha_scale']
        m_sun  = target_nml['phoenix']['m_sun']
        wltau  = target_nml['phoenix']['wltau']
        ngrrad  = target_nml['phoenix']['ngrrad']
        ieos  = target_nml['phoenix']['ieos']
        mixlng  = target_nml['phoenix']['mixlng']
        tau_conv_min  = target_nml['phoenix']['tau_conv_min']
        ieos = 5

        print(filename,':',teff,logg,m_sun,wltau,mixlng,tau_conv_min,zscale,alpha_scale, file=sys.stderr)

        if(Teff_new <=    0.): Teff_new = teff
        if(logg_new <= -999.): logg_new = logg

        m_sun_new = calcMass(Teff_new, logg_new)
        mixlng_new = calcMixingLength(Teff_new, logg_new)
        #print('old mass=',m_sun,' new mass=',m_sun_new,' rel. diff:',(m_sun-m_sun_new)/m_sun,file=sys.stderr)
        #print('old mixlng=',mixlng,' new mixlng=',mixlng_new,' rel. diff:',(mixlng-mixlng_new)/mixlng,file=sys.stderr)

        if(Teff_new != teff or logg_new != logg):
          print("old parameters: Teff, logg, mass, mixlng:",teff,logg,m_sun,mixlng,file=sys.stderr)
          print("new parameters: Teff, logg, mass, mixlng:",Teff_new,logg_new,m_sun_new,mixlng_new,file=sys.stderr)
          teff = Teff_new
          logg = logg_new
          m_sun = m_sun_new
          mixlng = mixlng_new

        #zscale = -3.5
        #print(filename,':',' new zscale=',zscale,file=sys.stderr)

#        alpha_scale = 0.2
#        print(filename,':',' new alpha_scale=',alpha_scale,file=sys.stderr)
#
# make patch:
# add more lines as needed!
#
        patch = {'phoenix': {}}
        patch['phoenix']['teff']= teff
        patch['phoenix']['r0']= r0
        patch['phoenix']['v0']= v0
        patch['phoenix']['zscale']= zscale
        patch['phoenix']['alpha_scale']= alpha_scale
        patch['phoenix']['m_sun']= m_sun
        patch['phoenix']['wltau']= wltau
        patch['phoenix']['ngrrad']= ngrrad
        patch['phoenix']['ieos']= ieos
        patch['phoenix']['mixlng']= mixlng
        patch['phoenix']['tau_conv_min']= tau_conv_min
        patch['phoenix']['logg']= logg

        # patch['phoenix']['eheu(04)']= eheu_Be
        # patch['phoenix']['nworkers']= nwrk
        # patch['phoenix']['nwlnodes']= nwl
        # patch['phoenix']['n_lin_mol_v']= nwrk
        # patch['phoenix']['n_lin_mol_g']= nwrk
        # patch['phoenix']['n_lin_atm_v']= nwrk
        # patch['phoenix']['n_lin_atm_g']= nwrk
        # patch['phoenix']['n_nlte_rates']= nwrk
        # patch['phoenix']['n_nlte_opac']= nwrk
        # patch['phoenix']['n_rt']= nwrk

#
# this if for ALL-IN:
#
#         if(teff < 5000): # for low temperature turn off high ions for these models
#           patch['phoenix']['nlte_level_number(073)']= 0  #  1104
#         if(teff < 4000): # for low temperature turn off high ions for these models
#           patch['phoenix']['nlte_level_number(007)']= 0  #  C IV
#           patch['phoenix']['nlte_level_number(011)']= 0  #  N IV
#           patch['phoenix']['nlte_level_number(015)']= 0  #  O IV
#           patch['phoenix']['nlte_level_number(026)']= 0  #  1304
#           patch['phoenix']['nlte_level_number(058)']= 0  #  1704
#           patch['phoenix']['nlte_level_number(086)']= 0  #   403
#           patch['phoenix']['nlte_level_number(090)']= 0  #   904
#           patch['phoenix']['nlte_level_number(106)']= 0  #  1804
#           patch['phoenix']['nlte_level_number(147)']= 0  #  1003
#           patch['phoenix']['nlte_level_number(148)']= 0  #  1004
#
# this is for the FULL setup
#
        if(teff < 3000): # for low temperature turn off high ions for these models
          patch['phoenix']['nlte_level_number(006)']= 0  #  K III
          patch['phoenix']['nlte_level_number(009)']= 0  # Mg III
          patch['phoenix']['nlte_level_number(012)']= 0  # Ca III
          patch['phoenix']['nlte_level_number(015)']= 0  # Na III
          patch['phoenix']['nlte_level_number(019)']= 0  #  C III
          patch['phoenix']['nlte_level_number(022)']= 0  #  N III
          patch['phoenix']['nlte_level_number(025)']= 0  #  O III
          patch['phoenix']['nlte_level_number(028)']= 0  # Si III
          patch['phoenix']['nlte_level_number(031)']= 0  # Fe III

#print("target patch:",patch)


#       need wider Voigt windows for cool models
        if(teff <= 2700.):
           patch['phoenix']['dlilam'] = 1000.0
           patch['phoenix']['dlimol'] = 50.0


#
# now we need to fix the job name, file names etc in the target.
# 0. construct file names etc. from the target_nml.
# 1. read the new file.
# 2. use the re engine to change the data in RAM
# 3. write the data back to the file.
#

#off    new_name =  'CN_NLTE_L=50e3Lsun_Teff='+f'{teff:.0f}'+'_v0='+f'{v0:.0f}'+'_z='+f'{zscale:0=+6.2f}'
#off        new_name =  'CN_NLTE_L=50e3Lsun_Teff='+f'{teff:.0f}'+'_v0='+f'{v0:.0f}'+'_z='+f'{zscale:0=+6.2f}'+'_Be='+f'{eheu_Be:0=+6.2f}'
        restart = re.sub('\\.20$','', filename.rstrip())
        #new_name = restart+'.Orosz'

        if(zscale != 0.0):
          if(alpha_scale == 0.0):
            job_name = 'lte'+f'{teff:0=5.0f}'+f'{-logg:3.2f}'+f'{zscale:0=+4.1f}'
          else:
            job_name = 'lte'+f'{teff:0=5.0f}'+f'{-logg:3.2f}'+f'{zscale:0=+4.1f}'+'.alpha='+f'{alpha_scale:0=+3.1f}'
        else:
          if(alpha_scale == 0.0):
            job_name = 'lte'+f'{teff:0=5.0f}'+f'{-logg:3.2f}'+'-'+f'{zscale:0=3.1f}'
          else:
            job_name = 'lte'+f'{teff:0=5.0f}'+f'{-logg:3.2f}'+'-'+f'{zscale:0=3.1f}'+'.alpha='+f'{alpha_scale:0=+3.1f}'

        new_name = job_name+'.FULL.PHOENIX-NewEra-SESAM-COND-2023.hummel2'

        new_name = re.sub('lte','nlte', new_name.rstrip())
        job_name = re.sub('lte','nlte', job_name.rstrip())

        newfile = new_name.strip()+file_stem
#
# patch these values back in the template:
#
        f90nml.patch(template, patch, newfile)

#
# now fix the names (etc) in the new dqs file:
#
        for zeile in fileinput.input(newfile,inplace=1):
            zeile = re.sub(' NAME=.*$',' NAME='+new_name, zeile.rstrip())
#off        zeile = re.sub(' RESTART=.*$',' RESTART='+restart+'.nobi', zeile.rstrip())
            zeile = re.sub(' RESTART=.*$',' RESTART='+restart, zeile.rstrip())
            #zeile = re.sub('job-name=.*$','job-name=lte'+f'{teff/1000.:.1f}'+'kK', zeile.rstrip())
            zeile = re.sub('job-name=.*$','job-name='+job_name, zeile.rstrip())
            #zeile = re.sub('tasks-per-node=.*$','tasks-per-node='+f'{n_MPI:d}', zeile.rstrip())
            if(zscale != 0.0):
              if(alpha_scale == 0.0):
                zeile = re.sub('z-0.0.AGSS.ACES.v19','z'+f'{zscale:0=+3.1f}'+'.AGSS.ACES.v19', zeile.rstrip())
              else:
                zeile = re.sub('z-0.0.AGSS.ACES.v19','z'+f'{zscale:0=+3.1f}'+'.alpha='+f'{alpha_scale:0=3.1f}'+'.AGSS.ACES.v19', zeile.rstrip())
            else:
              if(alpha_scale == 0.0):
                zeile = re.sub('z-0.0.AGSS.ACES.v19','z-'+f'{zscale:0=3.1f}'+'.AGSS.ACES.v19', zeile.rstrip())
              else:
                zeile = re.sub('z-0.0.AGSS.ACES.v19','z-'+f'{zscale:0=3.1f}'+'.alpha='+f'{alpha_scale:0=3.1f}'+'.AGSS.ACES.v19', zeile.rstrip())

            print(zeile)

        return newfile

#
# construct a restart file with no bi data (for this case)
# comment this out if the old bi are desired!
#

#off    outfile = open(restart+'.nobi.20','w')
#off    in_bi = False;
#off    with open(filename) as f:
#off       for line in f:
#off           realline = line.rstrip()
#off           if(in_bi):
#off               if(re.search('END departure coefficients',realline) is not None): in_bi = False;
#off           else:
#off               if(re.search('START departure coefficients',realline) is not None):
#off                   in_bi = True;
#off               else:
#off                   print(realline,file=outfile)
#off       outfile.close()

#
# run a test job:
#


n_MPI = -1
template = 'template.NewEraNLTE.SESAM.FULL.hummel2.dqs'
file_stem = '.array.dqs'

n_args = len(sys.argv)

if((sys.argv[1]).isdigit()):
 i = int(sys.argv[1])
# this is the 'do all in a row mode, active if argument >=0
 liste = sorted(glob.glob('nlte*FULL*PHOENIX-NewEra-SESAM-COND-2023.hummel2.20'))
 if(i >= len(liste)):
    sys.exit(-1)
 else:
    filename = liste[i]
else:
# single restart file mode:
    filename = sys.argv[1]

Teff_new = -1.0
logg_new = -999.

if(n_args > 2): Teff_new = float(sys.argv[2])
if(n_args > 3): logg_new = float(sys.argv[3])

print(filename, Teff_new, logg_new, file=sys.stderr)
product = update_dqs(filename,template,file_stem,n_MPI,Teff_new,logg_new)
print(product)
