import pyshark
import sys
import glob

directory = sys.argv[1]
extended = directory + '/'
paths = glob.glob(extended)

# vectorize pcaps to packets
all_features = []
all_feature_strings = []
protocol_frequencies = dict()
for path in paths:
    pcapPath = path + '/*.pcap'
    pcapFiles = glob.glob(pcapPath)
    # iterate over packets
    for pcapFile in pcapFiles:
        print("file")
        cap = pyshark.FileCapture(pcapFile, include_raw=True, use_json=True)
        for packet in cap:
            features = []
            features_strings = ""
            protocols = ["ARP", "LLC", "IP", "ICMP", "ICMPv6", "EAPOL", "TCP", "UDP", "HTTP", "HTTPS", "DHCP", "BOOTP", "SSDP", "DNS", "MDNS", "NTP"]
            for protocol in protocols:
                if protocol in packet:
                    features.append(1)
                    features_strings += "1"
                else:
                    features.append(0)
                    features_strings += "0"
            if hasattr(packet, 'tcp'):
                print(packet.tcp.srcport + ' -- ' + packet.tcp.dstport)
                if int(packet.tcp.srcport) > int(packet.tcp.dstport):
                    features_strings += str(packet.tcp.dstport) + ","
                    print(str(packet.tcp.dstport))
                else:
                    features_strings += str(packet.tcp.srcport)
                    print(str(packet.tcp.srcport))
            if hasattr(packet, 'udp'):
                print(packet.udp.srcport + ' -- ' + packet.udp.dstport)
                if int(packet.udp.srcport) > int(packet.udp.dstport):
                    features_strings += str(packet.udp.dstport) + ","
                    print(str(packet.udp.dstport))
                else:
                    features_strings += str(packet.udp.srcport)
                    print(str(packet.udp.srcport))
            protocol_frequencies[features_strings] = protocol_frequencies.get(features_strings, 0) + 1
            all_features.append(features)
            all_feature_strings.append(features_strings)

print(all_features)
print(list(set(all_feature_strings)))
print(len(set(all_feature_strings)))
print(protocol_frequencies)