import sys
from enum import IntEnum
import time
import clr
import matplotlib.pyplot as plt

from retrying import retry

from Parameters import *
Inst_p = Instrument_param


EXP_PATH = './Action_exp_setting/API_Test_loop/exper.info'
DATA_PATH = './Action_exp_setting/API_Test_loop/DataFile.data'

clr.AddReference("./Exp_Pyctrl_solartron/ModuLab_dll/ModuLabAPI")

from SolartronAnalytical.ModuLabAPI import ModuLabControl, NewResultsEventArgs


class EProgramMode(IntEnum):
    ECS = 0
    MTS = 1
    DSSC = 2


class ELookForResuls(IntEnum):
    Found = 0
    WrongSerial = 1
    WrongModel = 2
    WrongInstType = 3
    NotFound = 4
    NotOnSubnet = 5


class EQuantities(IntEnum):
    Time = 0
    Voltage = 1
    Current = 2
    Charge = 3
    Temperature = 4
    Power = 5
    Resistance = 6
    Capacitance = 7
    Frequency = 8
    Current_Density = 9
    Power_Density = 10
    Charge_Density = 11
    Range = 12
    LED_Intensity = 13
    LED_Power = 14


class EComplexQuantities(IntEnum):
    Impedance = 0
    Admittance = 1
    AC_Voltage = 2
    AC_Current = 3
    Permittivity = 4
    Rel_Permittivity = 5
    Modulus = 6
    Rel_Modulus = 7
    AC_Capacitance = 8
    Transfer_Function = 9


class EComplexComponent(IntEnum):
    Real = 0
    Imaginary = 1
    Magnitude = 2
    Phase = 3
    TanDelta = 4


class EChannels(IntEnum):
    Default = 0
    VoltageMain = 0
    VoltageAuxA = 1
    VoltageAuxB = 2
    VoltageAuxC = 3
    VoltageAuxD = 4
    CurrentForward = 0
    CurrentReverse = 1
    CurrentDifferential = 2
    TimeExperiment = 0
    TimeLoopStart = 1
    TimeStepStart = 2
    TemperatureSetPoint = 0
    TemperatureSample = 1
    TemperatureControl = 2
    RangeCurrent = 0
    RangeVoltage = 1
    RangeAuxA = 2
    RangeAuxB = 3
    RangeAuxC = 4
    RangeAuxD = 5


class EInstrumentStates(IntEnum):
    NotInUse = 0  # Not currently being used by modulab software
    InUse = 1  # Currently exclusively allocated to this copy of the software
    Measuring = 2  # Actually running an experiment
    Stopping = 3  # The software is trying to stop the instrument running
    Paused = 4  # The experiment is paused
    SafetyLimit = 5  # A safety limit was exceeded. Insterument in paused state
    MeasuringOCV = 6  # Measuring OCV running an experiment
    ExpEnd = 7  # The experiment has completed, but post-experiment actions are being run


# Global
# Holds all results
resultsAC = []
resultsDC = []
resultsDC_2 = []

'''
Class SolResult: 
-- A single measurement from the instrument This contains both DC and AC value, and allows calculate of derived values such as Power
GetSectionID: Get the ID of the section this result belongs to
GetSectionName: Get the Name of the section this result belongs to  
GetValue: Gets the the required value from the measurement If the value is not valid, NaN is returned  
GetValueCx: Gets the the required complex number value from the measurement If the value is not valid, NaN is returned    
'''


def PrintResultAC(result):
    print("{0:7.1f} {1:7.1f} {2:7.1f}".format(
        result.GetValue(EQuantities.Frequency, EChannels.Default),
        result.GetValueCx(EComplexQuantities.Impedance, EComplexComponent.Magnitude, EChannels.Default),
        result.GetValueCx(EComplexQuantities.Impedance, EComplexComponent.Phase, EChannels.Default)))

# for battery cycling
def PrintResultDC(result):
    print("{0:7.1f} {1:7.5f} {2:7.2f}".format(
        result.GetValue(EQuantities.Time, EChannels.Default),
        1000*result.GetValue(EQuantities.Current, EChannels.Default),  # Amp -> mA
        result.GetValue(EQuantities.Voltage, EChannels.Default),
        ))

def PlotAC(data):
    frequencies = list(map(lambda x: x.GetValue(EQuantities.Frequency, EChannels.Default), data))
    magnitudes = list(
        map(lambda x: x.GetValueCx(EComplexQuantities.Impedance, EComplexComponent.Magnitude, EChannels.Default), data))
    plt.loglog(frequencies, magnitudes)
    plt.show()


def InstStateChanged(source, args):
    print("Instrument State: {0}".format(EInstrumentStates(source.State).name))


def InstResultsAC(source, args):
    print("New Results")
    resultsAC.extend(args.Results)
    for res in args.Results:
        PrintResultAC(res)

def InstResultsDC(source, args):
    print("New Results")
    resultsDC.extend(args.Results)
    for res in args.Results:
        PrintResultDC(res)

def testAC():
    mc = ModuLabControl()
    mc.ProgramMode = EProgramMode.ECS

    # Virtual instrument instead of real (set to False to connect to the real instrument)
    mc.Virtual = True
    inst = mc.CreateInstrument("169.254.254.11", "169.254.254.10")
    inst.Connect()
    print(inst.GetInstrumentInfo())

    # Add event handlers
    inst.StateChanged += InstStateChanged
    inst.NewResults += InstResultsAC

    # start exp
    inst.StartExperiment(EXP_PATH, DATA_PATH)

    # wait for experiment to complete
    while EInstrumentStates(inst.State) != EInstrumentStates.InUse:
        time.sleep(1)

    # close connection to instrument
    inst.Disconnect()

    # plot results
    PlotAC(resultsAC)

    # close data files
    for datafile in mc.GetOpenDataFiles():
        datafile.Close()


###### Battery Part ######

def Battery_data_output(data,draw=False):
    times_list = list(map(lambda x: x.GetValue(EQuantities.Time, EChannels.Default), data))
    currents_list = list(map(lambda x: 1000*x.GetValue(EQuantities.Current, EChannels.Default), data))  # Amp -> mA
    voltages_list = list(map(lambda x: x.GetValue(EQuantities.Voltage, EChannels.Default), data))

    if draw is True:
        plt.plot(times_list, currents_list)
        plt.plot(times_list, voltages_list)
        plt.show()

    return times_list, currents_list, voltages_list


# Single core card. we will finish the code for multiple core card in the future.
@retry(stop_max_attempt_number=3)
def pycontrol_single(exppath,datapath,Virtual=True,mode=EProgramMode.ECS):
    """
    :param exppath: path to experiment to run
    :param datapath: Path to data folder
    :param (IPaddress): IP address of core card
    :param Virtual: Creates Simulated Virtual instruments instead of communicating with real instruments
    :param mode: Gets or sets the present program mode. ECS or MTS
    :return: times, currents, voltages
    """

    mc = ModuLabControl()
    mc.ProgramMode = mode

    # Virtual instrument instead of real (set to False to connect to the real instrument)
    mc.Virtual = Virtual
    IPs = Inst_p.IPaddress
    inst = mc.CreateInstrument(IPs[0],IPs[1])
    print(inst.LookFor("whether the instrument was found"))
    inst.Connect()
    print(inst.GetInstrumentInfo())

    # Add event handlers
    inst.StateChanged += InstStateChanged
    inst.NewResults += InstResultsDC

    # start exp
    inst.StartExperiment(exppath, datapath)
    # Start running the experiment and collect data in the given data file
    # Existing data folders will be overwritten

    # wait for experiment to complete
    while EInstrumentStates(inst.State) != EInstrumentStates.InUse:
        time.sleep(1)

    # close connection to instrument
    inst.Disconnect()

    # data output
    times, currents, voltages = Battery_data_output(resultsDC)

    # close data files
    for datafile in mc.GetOpenDataFiles():
        # GetOpenDataFiles(): Get a list of all open files
        datafile.Close()

    return times, currents, voltages


if __name__ == '__main__':
    # # test program
    # testAC()
    pycontrol_single(EXP_PATH, DATA_PATH)

