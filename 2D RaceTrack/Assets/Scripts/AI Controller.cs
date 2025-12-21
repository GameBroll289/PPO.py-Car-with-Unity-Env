using UnityEngine;

public class AICarController : MonoBehaviour
{
    public float speed;
    public float turnSpeed;
    private Rigidbody2D rb;

    void Awake()
    {
        // 1. Find the Rigidbody directly (Safest way)
        rb = GetComponent<Rigidbody2D>();
        
        // 2. Optional: Override settings from Car script if it exists
        Car car = GetComponent<Car>();
        if (car != null)
        {
            speed = car.speed;
            turnSpeed = car.turnSpeed;
        }
    }

    // REMOVED: FixedUpdate() 
    // REMOVED: MMF reading logic (The Master script handles this now)

    // ADDED: A public function that the Master script calls explicitly
    public void ManualMove(float move, float turn, float dt)
    {
        if (rb == null) return;

        // We use the passed 'dt' (0.02) instead of Time.deltaTime to be perfectly deterministic
        rb.AddForce(transform.up * (move * speed), ForceMode2D.Force);
        rb.AddTorque(-turn * turnSpeed * dt, ForceMode2D.Force);
    }
}