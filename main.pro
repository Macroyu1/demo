ProgramInfo
    Version = "S03.23"
    Time = "2024/10/8 13:14:25"
    RobotName = "IR-R220-270S5-E1N_01740513"
EndProgramInfo
Start;
    Byte LB1 = 0;
    Movj dot1,V[60],Z[0],Tool[0],Wobj[0];
    WaitInPos;
    While 1
        Movl dot2,V[60],Z[CP],Tool[0],Wobj[0];
        WaitInPos;
        Delay T[1];
        Movl dot1,V[60],Z[CP],Tool[0],Wobj[0];
        WaitInPos;
        Delay T[1];
        B[0] = B[0] + 1;
        If B[0] == 3
            Break;
        EndIf;
    EndWhile;
End;
