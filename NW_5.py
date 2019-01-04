import ns.core
import ns.csma
import ns.olsr
import ns.applications
import ns.wifi
import ns.mobility
import ns.internet
import ns.visualizer 
from numpy import random
import numpy as np 
import pandas as pd
random.seed(10)


# This example considers two hidden stations in an 802.11n network which supports MPDU aggregation.
# The user can specify whether RTS/CTS is used and can set the number of aggregated MPDUs.
#
# Example: ./waf --pyrun "examples/wireless/simple-ht-hidden-stations.py --enableRts=True --nMpdus=8"
#
# Network topology:
#

# 
#           +--------------------------------------------------------+
#           |                                                        |
#           |              802.11  Network: Central Controller       | 
#           |                                                        |
#           +--------------------------------------------------------+
#                    |       o o o(M AccessPoints (APs))       |
#                +--------+                               +--------+
#      wired LAN |  AP    |                     wired LAN |   AP   |
#     -----------|        |                    -----------|        |
#                ----------                               ----------
#                    |                                        |
#           +----------------+                       +----------------+
#           |     802.11     |                       |     802.11     |
#           |      net       |                       |       net      |
#           |   N-1 STAs     |                       |   N-1 STAs     |
#           +----------------+                       +----------------+
#



''' Wifi setup settings '''
class Wifi_setup():
    
    def __init__(self,maxAmpduSize):
        print("setup wifi phy and mac parameters")
        self.maxAmpduSize = maxAmpduSize;

    def wifi_phy(self,MCS,band,ap,ssid):
        self.ssid = ssid;
        channel =  ns.wifi.YansWifiChannelHelper.Default();
        channel.AddPropagationLoss("ns3::RangePropagationLossModel")
        #channel.AddPropagationLoss("ns3::FriisPropagationLossModel")
        self.phy = ns.wifi.YansWifiPhyHelper.Default()
        self.phy.SetPcapDataLinkType(ns.wifi.YansWifiPhyHelper.DLT_IEEE802_11_RADIO)
        self.phy.SetChannel(channel.Create())

        self.wifi = ns.wifi.WifiHelper()
        if band==5:
            self.wifi.SetStandard (ns.wifi.WIFI_PHY_STANDARD_80211n_5GHZ)
        else:
            self.wifi.SetStandard (ns.wifi.WIFI_PHY_STANDARD_80211n_2_4GHZ)
        
        #self.wifi.SetDeviceAttribute("DataRate", ns.core.StringValue("5Mbps"))

        self.wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", 
                                      "DataMode", ns.core.StringValue ("HtMcs"+str(MCS)),
                                      "ControlMode", ns.core.StringValue ("HtMcs0"))
        mac = ns.wifi.WifiMacHelper()
        mac.SetType ("ns3::ApWifiMac",
            "Ssid", ns.wifi.SsidValue (self.ssid),
            "EnableBeaconJitter", ns.core.BooleanValue (False),
            "BE_MaxAmpduSize", ns.core.UintegerValue (self.maxAmpduSize))
        AP =  ns.network.NetDeviceContainer()
        AP =  self.wifi.Install(self.phy,mac,ap)
        return AP,self.phy;

    def link_sta(self,station):
        mac = ns.wifi.WifiMacHelper()
        mac.SetType ("ns3::StaWifiMac",
                "Ssid", ns.wifi.SsidValue (self.ssid),
                "ActiveProbing", ns.core.BooleanValue (False),
                "BE_MaxAmpduSize", ns.core.UintegerValue(self.maxAmpduSize))
        sta = ns.network.NetDeviceContainer()
        sta = self.wifi.Install(self.phy,mac,station) 
        return sta,self.phy;


''' Main function to create network '''
def main(argv):
    cmd = ns.core.CommandLine()
    cmd.payloadSize =  1472 
    cmd.simulationTime = 5;
    cmd.nMpdus =  1;
    cmd.maxAmpduSize = 0;
    cmd.enableRts = False
    cmd.minExpectedThroughput = 0;
    cmd.maxExpectedThroughput = 0;
    cmd.nAPs = 9; 
    cmd.nSTAs = 27; 
    cmd.area = 50;
    cmd.MCS = 7;

    cmd.AddValue ("nMpdus", "Number of aggregated MPDUs")
    cmd.AddValue ("payloadSize", "Payload size in bytes")
    cmd.AddValue ("enableRts", "Enable RTS/CTS")
    cmd.AddValue ("simulationTime", "Simulation time in seconds")
    cmd.AddValue ("minExpectedThroughput", "if set, simulation fails if the lowest throughput is below this value")
    cmd.AddValue ("maxExpectedThroughput", "if set, simulation fails if the highest throughput is above this value")
    cmd.AddValue("nAPs","Number of APs")
    cmd.AddValue("nSTAs","Number of STAs")
    cmd.AddValue("area","Network area")
    cmd.AddValue("MCS","MCS value for al APs")
    cmd.Parse (sys.argv)

    payloadSize = int (cmd.payloadSize)
    simulationTime = float (cmd.simulationTime)
    nMpdus = int (cmd.nMpdus)
    maxAmpduSize = int (cmd.maxAmpduSize)
    enableRts = cmd.enableRts
    minExpectedThroughput = cmd.minExpectedThroughput
    maxExpectedThroughput = cmd.maxExpectedThroughput
    nAPs = int(cmd.nAPs);
    nSTAs= int (cmd.nSTAs);
    area = int(cmd.area);
    MCS =  int(cmd.MCS);
    
    band = 5; # 5GHz band
    if enableRts:
        ns.core.Config.SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold", 
                                   ns.core.StringValue ("999999"))
    else:
        ns.core.Config.SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold", 
                                   ns.core.StringValue ("0"))    

    ns.core.Config.SetDefault ("ns3::WifiRemoteStationManager::FragmentationThreshold", 
                               ns.core.StringValue ("990000"))


    # Set the maximum size for A-MPDU with regards to the payload size
    maxAmpduSize = nMpdus * (payloadSize + 200)

    # Set the maximum wireless range to 5 meters in order to reproduce a hidden node scenario
    ns.core.Config.SetDefault("ns3::RangePropagationLossModel::MaxRange", ns.core.DoubleValue (20))
    
    wifiStaNodes = ns.network.NodeContainer ()
    wifiStaNodes.Create (nSTAs)

    wifiApNode = ns.network.NodeContainer ()
    wifiApNode.Create (nAPs)

    ''' Add mobility condition for APs and STAs '''
    mobility = ns.mobility.MobilityHelper ()
    positionAlloc = ns.mobility.ListPositionAllocator ()


    ''' Import data from the CSV file about AP location , Station location and STation maps '''
    inputs   =  pd.read_csv('inputs.csv')
    AP_xloc  =  inputs['AP_xloc']
    AP_yloc  =  inputs['AP_yloc']
    STA_xloc =  inputs['STA_xloc']
    STA_yloc =  inputs['STA_yloc']
    STA_map  =  inputs['STA_map']
    UL_R     =  inputs['UL_Rate']
    DL_R     =  inputs['DL_Rate']


    ''' mobility pattern for APs: considering 9 APs '''
    for i in range(nAPs):
        positionAlloc.Add (ns.core.Vector(AP_xloc[i],AP_yloc[i],0))

    ''' mobility pattern for STAs , randomly disbtributed STAs '''
    mobility.SetPositionAllocator (positionAlloc)
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install (wifiApNode)

    positionAlloc1 = ns.mobility.ListPositionAllocator ()
    for j in range(nSTAs):
        positionAlloc1.Add(ns.core.Vector(STA_xloc[j],STA_yloc[j],0))

    mobility.SetPositionAllocator (positionAlloc1)
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel")
    mobility.Install (wifiStaNodes)


    ''' set up ipv4 address  '''
    address = ns.internet.Ipv4AddressHelper()
    address.SetBase (ns.network.Ipv4Address ("192.168.1.0"), ns.network.Ipv4Mask ("255.255.255.0"))
    stack = ns.internet.InternetStackHelper ()
    stack.Install (wifiApNode)
    stack.Install (wifiStaNodes)

    ''' create dictionary to add APs and STAs connection info '''
    AP ={};
    STA = {}
    ApInterface  = {};
    StaInterface  = {};

    ''' assuming client data is sent to every STA from their respective APs  '''
    port = 9;  
    server = ns.applications.UdpServerHelper (port)
    serverAp = server.Install(wifiApNode)# (ns.network.NodeContainer(wifiApNode.Get(i)))
    serverAp.Start (ns.core.Seconds (0.0))
    serverAp.Stop (ns.core.Seconds (simulationTime + 1))

    ''' assuming STA is the server in this section '''
    port1 = 90;  
    server1 = ns.applications.UdpServerHelper (port1)
    serverSta = server1.Install(wifiStaNodes)# (ns.network.NodeContainer(wifiApNode.Get(i)))
    serverSta.Start (ns.core.Seconds (0.0))
    serverSta.Stop (ns.core.Seconds (simulationTime + 1))

    ''' setup wifi parameters for APs and STAs '''
    WIFI =  Wifi_setup(maxAmpduSize) # call  wifi setup class
    if abs(MCS)==MCS:
        print('== Network with Similar AP capacities ==')
        MCS =   np.array([MCS]*nAPs)#
    else:
        print('== Network with different AP capacities ==')
        MCS = random.randint(8,size=nAPs) # generate different MCS values for different APs 
        print('MCS for APs: {}'.format(MCS))

    ''' build connection between APs and STAs '''
    for i in range(nAPs):
        print(" building network for AP-{}".format(i+1));
        ssid = ns.wifi.Ssid("AP-"+str(i+1))
        AP[str(i)],Phy =  WIFI.wifi_phy(MCS[i],band,wifiApNode.Get(i),ssid)#wifi.Install (phy, mac, wifiApNode.Get(i))
        ApInterface[str(i)] = ns.internet.Ipv4InterfaceContainer ()
        ApInterface[str(i)] = address.Assign (AP[str(i)])

        k, = np.where(STA_map==i) # m corresponds to row and k corresponds to column position

        ''' create client application for ap where AP is a server'''
        client = ns.applications.UdpClientHelper (ApInterface[str(i)].GetAddress (0), port) 
        Phy.EnablePcap ("SimpleHtHiddenStations_py_Ap_"+str(i),ns.network.NodeContainer 
                        (wifiApNode.Get(i)) )

        for j in k: # establish connection with AP-i and STA-j as predicted in the algorithm
            STA[str(j)],Phy = WIFI.link_sta(wifiStaNodes.Get(j)) 
            StaInterface[str(j)] = ns.internet.Ipv4InterfaceContainer()
            StaInterface[str(j)] = address.Assign(STA[str(j)])
            if UL_R[j]>0:
                UL_pack = (UL_R[j]*8)/payloadSize
                client.SetAttribute ("MaxPackets", ns.core.UintegerValue (UL_pack))
                client.SetAttribute ("Interval", ns.core.TimeValue (ns.core.Time ("0.0002")))
                client.SetAttribute ("PacketSize", ns.core.UintegerValue (payloadSize))
                clientSta = client.Install (ns.network.NodeContainer (wifiStaNodes.Get(j)))
                clientSta.Start(ns.core.Seconds(1.0))
                clientSta.Stop(ns.core.Seconds(simulationTime + 1))

            if DL_R[j]>0:
                DL_pack = (DL_R[j]*8)/payloadSize
                client2 = ns.applications.UdpClientHelper(StaInterface[str(j)].GetAddress (0), port1)
                client2.SetAttribute ("MaxPackets", ns.core.UintegerValue (DL_pack))
                client2.SetAttribute ("Interval", ns.core.TimeValue (ns.core.Time  
                                                                     ("0.0002")))#
                client2.SetAttribute ("PacketSize", ns.core.UintegerValue (payloadSize))
                clientAp = client2.Install (ns.network.NodeContainer (wifiApNode.Get(i)))
                clientAp.Start (ns.core.Seconds (1.0))
                clientAp.Stop (ns.core.Seconds (simulationTime + 1))
                Phy.EnablePcap ("SimpleHtHiddenStations_py_Sta"+str(j),             
                                ns.network.NodeContainer(wifiStaNodes.Get(j)))
                
    ''' run simulation '''
    ns.core.Simulator.Stop(ns.core.Seconds (simulationTime + 1))
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()

           
    ''' Upload and Download Throughput calculation '''
    
    ''' Upload throughput determined at AP side '''
    UL_th =[]; DL_th = []
    for i in range(nAPs):
        totalPacketsThrough = serverAp.Get(i).GetReceived()
        U_th = totalPacketsThrough * payloadSize * 8 / (simulationTime * 1000000.0)
        UL_th.append(U_th)
    
    UL_ap = np.zeros(shape=nAPs)
    ''' Download throughput determined at STA side '''
    for j in range(nSTAs):
        totalPacketsThrough = serverSta.Get(j).GetReceived()
        D_th = totalPacketsThrough * payloadSize * 8 / (simulationTime * 1000000.0)
        UL_ap[int(STA_map[j])] = UL_ap[int(STA_map[j])] + D_th; 
        DL_th.append(D_th)
        
    ''' Store it in pandas format and write the data to a csv file '''
    res = {
           'U_Throughput_AP': pd.Series(UL_th),
           'D_Throughput_AP': pd.Series(UL_ap)
          }
#            , 'D_Throughput': pd.Series(DL_th)
#           }

    df = pd.DataFrame(res)
    df.to_csv('NW_th.csv') # this file is saved in the directory where waf and 
    
    print('=========================================================')
    print('===== Finished Simulation for the current time slot =====')#scratch file are located   
    print('=========================================================')
    
    return 0

 
if __name__ == '__main__':
    import sys
    sys.exit (main (sys.argv))

