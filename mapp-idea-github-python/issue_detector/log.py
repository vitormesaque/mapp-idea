import sys
import os

class Log:

    outdir = None

    def __init__(self,outdir):
      self.outdir = outdir

    def write_step(self,log):
        f = open(self.outdir + 'sys_output_step.log', 'w')
        f.write(str(log))
        f.close()

    def write_console(self,log):
        f = open(self.outdir + 'sys_output.log', 'w')
        f.write(str(log))
        f.close()

    def write_status(self,log):
        f = open(self.outdir + 'sys_status.log', 'w')
        f.write(log)
        f.close()
