using UnityEngine;
using System.IO.MemoryMappedFiles;

namespace two{
public class Player : MonoBehaviour
{
    private const float SIGNAL_CODE = -999.0f;
    const string memoryName = "unity_ram2";
    const int slotCount = 30;   // must match Python
    const int slotSize = 4;     // float32
    const int totalSize = slotCount * slotSize;

    MemoryMappedFile mmf;
    MemoryMappedViewAccessor accessor;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        mmf = MemoryMappedFile.CreateOrOpen(memoryName, totalSize, MemoryMappedFileAccess.ReadWrite);
        accessor = mmf.CreateViewAccessor(0, totalSize, MemoryMappedFileAccess.ReadWrite);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        float vertical = ReadSlot(0);
        float horizontal = ReadSlot(1);

        // IF PYTHON HASN'T SENT A NEW COMMAND, WAIT.
        if (vertical == SIGNAL_CODE) return;
        transform.Translate(Vector3.up * vertical * 5f);
        transform.Rotate(Vector3.forward, -horizontal * 100f);

        // TELL PYTHON: "I FINISHED THIS FRAME"
        WriteFloat(0, SIGNAL_CODE);
    }

    float ReadSlot(int index)
    {
        float value;
        accessor.Read(index * slotSize, out value);
        return value;
    }

    void OnApplicationQuit()
    {
        accessor?.Dispose();
        mmf?.Dispose();
    }

    void WriteFloat(int slot, float value)
    {
        accessor.Write(slot * slotSize, value);
    }
}
}