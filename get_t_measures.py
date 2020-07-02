
function [t_measures] = get_t_measures(mouse)

  %%% put in here as well position around gate and position around reward, probability of reward reception, etc
  t_measures = [];
  
  if ismember(mouse,["879","882","884","886"])
    t_measures(1) = 0;
    t_measures(2) = 4;
    t_measures(3) = 24;
    t_measures(4) = 28;
    t_measures(5) = 48;
    t_measures(6) = 52;
    t_measures(7) = 120;   %% weekend!
    t_measures(8) = 124;
    t_measures(9) = 148;
    t_measures(10) = 168;
    t_measures(11) = 172;
    t_measures(12) = 192;
    t_measures(13) = 196;
    t_measures(14) = 216;
    t_measures(15) = 220;
    t_measures(16) = 292;
    t_measures(17) = 293;
    t_measures(18) = 316;
    t_measures(19) = 336;
    t_measures(20) = 340;
    t_measures(21) = 360;
    t_measures(22) = 364;
    t_measures(23) = 384;
    t_measures(24) = 388;
  
  elseif ismember(mouse,["839","840","841"])
    
    t_measures(1) = 0;
    t_measures(2) = 20;
    t_measures(3) = 24;
    t_measures(4) = 44;
    t_measures(5) = 48;
    t_measures(6) = 68;
    t_measures(7) = 72;
    t_measures(8) = 140;
    t_measures(9) = 144;
    t_measures(10) = 168;
    t_measures(11) = 188;
    t_measures(12) = 192;
    t_measures(13) = 212;
    t_measures(14) = 216;
    t_measures(15) = 236;
    t_measures(16) = 240;
    t_measures(17) = 336;
    t_measures(18) = 337;
    t_measures(19) = 356;
    t_measures(20) = 360;
    t_measures(21) = 380;
    t_measures(22) = 384;
    t_measures(23) = 404;
    t_measures(24) = 408;
    
  elseif ismember(mouse,["34","35"])
    
    t_measures(1) = 0;
    t_measures(2) = 4;
    t_measures(3) = 24;
    t_measures(4) = 28;
    t_measures(5) = 48;
    t_measures(6) = 52;
    t_measures(7) = 120;
    t_measures(8) = 124;
    t_measures(9) = 148;
    t_measures(10) = 168;
    t_measures(11) = 172;
    t_measures(12) = 192;
    t_measures(13) = 196;
    t_measures(14) = 216;
    t_measures(15) = 220;
    t_measures(16) = 221;
    t_measures(17) = 288;
    t_measures(18) = 292;
    t_measures(19) = 312;
    t_measures(20) = 316;
    t_measures(21) = 336;
    t_measures(22) = 340;
    
  elseif ismember(mouse,["243"])
    
    t_measures(1) = 0;
    t_measures(2) = 20;
    t_measures(3) = 24;
    t_measures(4) = 44;
    t_measures(5) = 68;
    t_measures(6) = 92;
    t_measures(7) = 96;
    t_measures(8) = 116;
    t_measures(9) = 120;
    t_measures(10) = 140;
    t_measures(11) = 144;
    t_measures(12) = 164;
    t_measures(13) = 168;
    t_measures(14) = 236;
    t_measures(15) = 240;
    t_measures(16) = 260;
    t_measures(17) = 264;
    t_measures(18) = 284;
    t_measures(19) = 288;
    t_measures(20) = 308;
    t_measures(21) = 312;
    t_measures(22) = 332;
    t_measures(23) = 336;
    t_measures(24) = 404;
    t_measures(25) = 408;
    t_measures(26) = 428;
    t_measures(27) = 432;
    t_measures(28) = 452;
    t_measures(29) = 456;
    t_measures(30) = 476;
    t_measures(31) = 480;
    t_measures(32) = 500;
    t_measures(33) = 504;
    t_measures(34) = 572;
    t_measures(35) = 576;
    t_measures(36) = 596;
    t_measures(37) = 600;
    t_measures(38) = 620;
    t_measures(39) = 624;
    t_measures(40) = 644;
    t_measures(41) = 980;
    t_measures(42) = 1004;
    t_measures(43) = 1008;
    t_measures(44) = 1076;
    t_measures(45) = 1080;
    t_measures(46) = 1100;
    t_measures(47) = 1104;
    t_measures(48) = 1124;
    t_measures(49) = 1148;
    t_measures(50) = 1152;
    t_measures(51) = -604;
    t_measures(52) = -603;
    t_measures(53) = -602;
    t_measures(54) = -672;
    t_measures(55) = -671;
    t_measures(56) = -670;
    t_measures(57) = -912;
    t_measures(58) = -911;
    t_measures(59) = -910;
    t_measures(60) = -792;
    t_measures(61) = -791;
    t_measures(62) = -790;
    t_measures(63) = -840;
    t_measures(64) = -839;
    t_measures(65) = -838;
    t_measures(66) = -940;
    t_measures(67) = -939;
    t_measures(68) = -938;
    t_measures(69) = -1008;
    t_measures(70) = -1007;
    t_measures(71) = -1006;
    
  elseif ismember(mouse,["244","245","246"])
    
    %% those are not right!
    
    para.t_s(1) = 0;     %% 01.16. AM
    para.t_s(2) = 4;     %% 01.16. PM
    para.t_s(3) = 72;     %% 01.19. AM
    para.t_s(4) = 96;     %% 01.20. AM
    para.t_s(5) = 100;     %% 01.20. PM
    para.t_s(6) = 120;     %% 01.21. AM
    para.t_s(7) = 124;     %% 01.21. PM
    para.t_s(8) = 144;     %% 01.22. AM
    para.t_s(9) = 148;     %% 01.22. PM
    para.t_s(10) = 168;     %% 01.23. AM
    para.t_s(11) = 172;     %% 01.23. PM
    para.t_s(12) = 240;     %% 01.26. AM
    para.t_s(13) = 244;     %% 01.26. PM
    para.t_s(14) = 264;     %% 01.27. AM
    para.t_s(15) = 268;     %% 01.27. PM
    para.t_s(16) = 288;     %% 01.28. AM
    para.t_s(17) = 292;     %% 01.28. PM
    para.t_s(18) = 312;     %% 01.29. AM
    para.t_s(19) = 316;     %% 01.29. PM
    para.t_s(20) = 336;     %% 01.30. AM
    para.t_s(21) = 340;     %% 01.30. PM
    para.t_s(22) = 408;     %% 02.02. AM
    para.t_s(23) = 412;     %% 02.02. PM
    para.t_s(24) = 432;     %% 02.03. AM
    para.t_s(25) = 436;     %% 02.03. PM
    para.t_s(26) = 456;     %% 02.04. AM
    para.t_s(27) = 460;     %% 02.04. PM
    para.t_s(28) = 480;     %% 02.05. AM
    para.t_s(29) = 484;     %% 02.05. PM
    para.t_s(30) = 504;     %% 02.06. AM
    
  elseif ismember(mouse,["918shKO","943shKO"])
    
    t_measures(1) = 0;
    t_measures(2) = 4;
    t_measures(3) = 24;
    t_measures(4) = 28;
    t_measures(5) = 48;
    t_measures(6) = 52;
    t_measures(7) = 72;
    t_measures(8) = 76;
    t_measures(9) = 96;
    t_measures(10) = 100;
    t_measures(11) = 120;
    t_measures(12) = 168;
    t_measures(13) = 172;
    t_measures(14) = 192;
    t_measures(15) = 196;
    t_measures(16) = 197;
    t_measures(17) = 216;
    t_measures(18) = 240;
    t_measures(19) = 244;
    t_measures(20) = 264;
    t_measures(21) = 268;
    t_measures(22) = 288;
    t_measures(23) = 312;
    t_measures(24) = 316;
    t_measures(25) = 336;
    t_measures(26) = 340;
    t_measures(27) = 360;
    t_measures(28) = 364;
    
  elseif ismember(mouse,["549","551"])
    t_measures(1) = 0;
    t_measures(2) = 4;
    t_measures(3) = 24;
    t_measures(4) = 28;
    t_measures(5) = 48;
    t_measures(6) = 52;
    t_measures(7) = 72;
    t_measures(8) = 76;
    t_measures(9) = 96;
    t_measures(10) = 100;
    t_measures(11) = 172;
    t_measures(12) = 192;
    t_measures(13) = 196;
    t_measures(14) = 216;
    t_measures(15) = 220;
    t_measures(16) = 221;
    t_measures(17) = 240;
    t_measures(18) = 244;
    t_measures(19) = 264;
    t_measures(20) = 268;
    t_measures(21) = 1416;
    t_measures(22) = 1420;
    t_measures(23) = 1440;
    t_measures(24) = 1444;
    t_measures(25) = 1464;
    t_measures(26) = 1468;
    t_measures(27) = 1488;
    t_measures(28) = 1492;
  
  elseif ismember(mouse,["756","757","758"])
    
    t_measures(1) = 0;
    t_measures(2) = 4;
    t_measures(3) = 24;
    t_measures(4) = 28;
    t_measures(5) = 48;
    t_measures(6) = 52;
    t_measures(7) = 72;
    t_measures(8) = 76;
    t_measures(9) = 96;
    t_measures(10) = 100;
    t_measures(11) = 172;
    t_measures(12) = 192;
    t_measures(13) = 196;
    t_measures(14) = 216;
    t_measures(15) = 220;
    t_measures(16) = 240;
    t_measures(17) = 241;
    t_measures(18) = 244;
    t_measures(19) = 264;
    t_measures(20) = 268;
    if ismember(mouse,["756"])
      t_measures(21) = 288;
      t_measures(22) = 292;
      t_measures(23) = 1080;
      t_measures(24) = 1084;
      t_measures(25) = 1104;
      t_measures(26) = 1108;
      t_measures(27) = 1128;
      t_measures(28) = 1132;
      t_measures(29) = 1152;
      t_measures(30) = 1156;
    elseif ismember(mouse,["757","758"])
      t_measures(21) = 912;
      t_measures(22) = 916;
      t_measures(23) = 936;
      t_measures(24) = 940;
      t_measures(25) = 960;
      t_measures(26) = 964;
      t_measures(27) = 984;
      t_measures(28) = 988;
    end
    
  elseif ismember(mouse,["65","66","72"])
    t_measures(1) = 0;
    t_measures(2) = 4;
    t_measures(3) = 5;
    t_measures(4) = 24;
    t_measures(5) = 28;
    t_measures(6) = 48;
    t_measures(7) = 52;
    t_measures(8) = 72;
    t_measures(9) = 76;
    t_measures(10) = 96;
    t_measures(11) = 100;
    t_measures(12) = 120;
    t_measures(13) = 124;
    t_measures(14) = 144;
    t_measures(15) = 148;
    t_measures(16) = 192;
    t_measures(17) = 193;
    t_measures(18) = 196;
    t_measures(19) = 216;
    t_measures(20) = 220;
    t_measures(21) = 221;
    t_measures(22) = 240;
    t_measures(23) = 244;
    t_measures(24) = 264;
    t_measures(25) = 265;
    t_measures(26) = 268;
    t_measures(27) = 360;
    t_measures(28) = 364;
    t_measures(29) = 365;
    t_measures(30) = 366;
    t_measures(31) = 384;
    t_measures(32) = 388;
    t_measures(33) = 389;
    t_measures(34) = 432;
    t_measures(35) = 436;
    t_measures(36) = 437;
    
    %% somewhere in here, m66 has an additional session...
%      t_measures(36) = 456;
%      t_measures(37) = 460;
%      t_measures(38) = 461;
    
    
    
    
    
    
    
  elseif ismember(mouse,["762"])
    t_measures(1) = 0;
    t_measures(2) = 20;
    t_measures(3) = 24;
    t_measures(4) = 44;
    t_measures(5) = 48;
    t_measures(6) = 68;
    t_measures(7) = 72;
    t_measures(8) = 140;
    t_measures(9) = 144;
    t_measures(10) = 168;
    t_measures(11) = 188;
    t_measures(12) = 192;
    t_measures(13) = 212;
    t_measures(14) = 216;
    t_measures(15) = 236;
    t_measures(16) = 240;
    t_measures(17) = 312;
    t_measures(18) = 336;
    t_measures(19) = 360;
    t_measures(20) = 384;
    t_measures(21) = 408;
    t_measures(22) = 528;
    t_measures(23) = 552;
    t_measures(24) = 576;
    t_measures(25) = 840;
    t_measures(26) = 864;
    t_measures(27) = 888;
    t_measures(28) = 912;
    t_measures(29) = 1008;
    t_measures(30) = 1028;
    t_measures(31) = 1032;
    t_measures(32) = 1052;
    t_measures(33) = 1056;
    t_measures(34) = 1076;
    t_measures(35) = 1080;
    t_measures(36) = 1148;
    t_measures(37) = 1152;
    t_measures(38) = 1172;
    t_measures(39) = 1176;
    t_measures(40) = 1196;
    t_measures(41) = 1200;
    t_measures(42) = 1220;
    t_measures(43) = 1224;
    t_measures(44) = 1244;
    t_measures(45) = 1248;
    t_measures(46) = 1316;
    t_measures(47) = 1320;
    t_measures(48) = 1340;
    t_measures(49) = 1364;
    t_measures(50) = 1368;
    t_measures(51) = 1388;
    t_measures(52) = 1392;
    t_measures(53) = 1412;
    t_measures(54) = 1416;
    t_measures(55) = 1484;
    t_measures(56) = 1488;
    t_measures(57) = 1512;
    t_measures(58) = 1532;
    t_measures(59) = 1536;
    t_measures(60) = 1556;
    t_measures(61) = 1560;
    t_measures(62) = 1580;
    t_measures(63) = 1584;
    t_measures(64) = 1700;
    t_measures(65) = 1704;
    t_measures(66) = 1724;
    t_measures(67) = 1728;
    t_measures(68) = 1748;
    t_measures(69) = 1752;
    t_measures(70) = 1820;
    t_measures(71) = 1824;
    t_measures(72) = 1844;
    t_measures(73) = 1848;
    t_measures(74) = 1868;
    t_measures(75) = 1872;
    t_measures(76) = 1892;
    t_measures(77) = 1896;
    t_measures(78) = 1916;
    t_measures(79) = 1920;
    t_measures(80) = 1988;
    t_measures(81) = 1992;
    t_measures(82) = 2012;
    t_measures(83) = 2016;
    t_measures(84) = 2036;
    t_measures(85) = 2040;
    t_measures(86) = 2060;
    t_measures(87) = 2064;
  end
end